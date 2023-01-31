import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmflow.geometry import forward_backward_consistency_check
from .util import MyPReLU

backwarp_tenGrid = {}


def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3], dtype=tenIn.dtype, device=tenIn.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2], dtype=tenIn.dtype, device=tenIn.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1)
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)


torch.fx.wrap('backwarp')
torch.fx.wrap('forward_backward_consistency_check')


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_net = nn.Sequential(
            nn.Conv2d(14, 64, 3, 1, 1),
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            MyPReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10):
        metric0 = F.l1_loss(img0, backwarp(img1, flow01), reduction='none').mean([1], True)
        metric1 = F.l1_loss(img1, backwarp(img0, flow10), reduction='none').mean([1], True)
        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        metric = self.metric_net(torch.cat((img0, -metric0, flow01, fwd_occ.unsqueeze(1), img1, -metric1, flow10, bwd_occ.unsqueeze(1)), 1))

        return metric[:, :1], metric[:, 1:2]
