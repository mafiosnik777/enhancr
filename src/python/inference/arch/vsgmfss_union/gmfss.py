import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FusionNet import AnimeInterp
from .gmflow.gmflow import GMFlow
from .IFNet_HDv3 import IFNet
from .MetricNet import MetricNet


class GMFSS(nn.Module):
    def __init__(self, model_dir, model_type, scale, ensemble):
        super(GMFSS, self).__init__()
        self.flownet = GMFlow()
        self.ifnet = IFNet(ensemble)
        self.metricnet = MetricNet()
        self.fusionnet = AnimeInterp()
        self.flownet.load_state_dict(torch.load(os.path.join(model_dir, 'flownet.pkl'), map_location='cpu'))
        self.ifnet.load_state_dict(torch.load(os.path.join(model_dir, 'rife.pkl'), map_location='cpu'))
        self.metricnet.load_state_dict(torch.load(os.path.join(model_dir, f'metric_{model_type}.pkl'), map_location='cpu'))
        self.fusionnet.load_state_dict(torch.load(os.path.join(model_dir, f'fusionnet_{model_type}.pkl'), map_location='cpu'))
        self.scale = scale

    def reuse(self, img0, img1):
        feat11, feat12, feat13 = self.fusionnet.feat_ext((img0 - 0.5) / 0.5)
        feat21, feat22, feat23 = self.fusionnet.feat_ext((img1 - 0.5) / 0.5)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")

        if self.scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=self.scale, mode="bilinear")
            imgf1 = F.interpolate(img1, scale_factor=self.scale, mode="bilinear")
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if self.scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor=1. / self.scale, mode="bilinear") / self.scale
            flow10 = F.interpolate(flow10, scale_factor=1. / self.scale, mode="bilinear") / self.scale

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def forward(self, img0, img1, timestep):
        reuse_things = self.reuse(img0, img1)

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")
        timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear")

        merged = self.ifnet(img0, img1, timestep)

        return self.fusionnet(img0, img1, reuse_things, merged, timestep)
