import torch
import torch.nn as nn
import torch.nn.functional as F

from .softsplat import softsplat as warp
from .util import MyPixelShuffle, MyPReLU


# Residual Block
def ResidualBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        MyPReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    )


# downsample block
def DownsampleBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        MyPReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )


# upsample block
def UpsampleBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        MyPReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
    )


class PixelShuffleBlcok(nn.Module):
    def __init__(self, in_feat, num_feat, num_out_ch):
        super(PixelShuffleBlcok, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_feat, num_feat, 3, 1, 1),
            MyPReLU()
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            MyPixelShuffle(2)
        )
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x

# grid network
class GridNet(nn.Module):
    def __init__(self, in_channels, in_channels1, in_channels2, in_channels3, out_channels):
        super(GridNet, self).__init__()

        self.residual_model_head = ResidualBlock(in_channels, 32, stride=1)
        self.residual_model_head1 = ResidualBlock(in_channels1, 32, stride=1)
        self.residual_model_head4 = ResidualBlock(in_channels//2, 32, stride=1)
        self.residual_model_head2 = ResidualBlock(in_channels2, 64, stride=1)
        self.residual_model_head3 = ResidualBlock(in_channels3, 96, stride=1)
        self.residual_model_01=ResidualBlock(32, 32, stride=1)
        #self.residual_model_02=ResidualBlock(32, 32, stride=1)
        #self.residual_model_03=ResidualBlock(32, 32, stride=1)
        self.residual_model_04=ResidualBlock(32, 32, stride=1)
        self.residual_model_05=ResidualBlock(32, 32, stride=1)
        self.residual_model_tail=PixelShuffleBlcok(32, 32, out_channels)


        self.residual_model_11=ResidualBlock(64, 64, stride=1)
        #self.residual_model_12=ResidualBlock(64, 64, stride=1)
        #self.residual_model_13=ResidualBlock(64, 64, stride=1)
        self.residual_model_14=ResidualBlock(64, 64, stride=1)
        self.residual_model_15=ResidualBlock(64, 64, stride=1)

        self.residual_model_21=ResidualBlock(96, 96, stride=1)
        #self.residual_model_22=ResidualBlock(96, 96, stride=1)
        #self.residual_model_23=ResidualBlock(96, 96, stride=1)
        self.residual_model_24=ResidualBlock(96, 96, stride=1)
        self.residual_model_25=ResidualBlock(96, 96, stride=1)

        #

        self.downsample_model_10=DownsampleBlock(32, 64, stride=2)
        self.downsample_model_20=DownsampleBlock(64, 96, stride=2)

        self.downsample_model_11=DownsampleBlock(32, 64, stride=2)
        self.downsample_model_21=DownsampleBlock(64, 96, stride=2)

        #self.downsample_model_12=DownsampleBlock(32, 64, stride=2)
        #self.downsample_model_22=DownsampleBlock(64, 96, stride=2)

        #

        #self.upsample_model_03=UpsampleBlock(64, 32, stride=1)
        #self.upsample_model_13=UpsampleBlock(96, 64, stride=1)

        self.upsample_model_04=UpsampleBlock(64, 32, stride=1)
        self.upsample_model_14=UpsampleBlock(96, 64, stride=1)

        self.upsample_model_05=UpsampleBlock(64, 32, stride=1)
        self.upsample_model_15=UpsampleBlock(96, 64, stride=1)

    def forward(self, x, x1, x2, x3, x4):
        X00=self.residual_model_head(x) + self.residual_model_head1(x1) + self.residual_model_head4(x4)      #---   182 ~ 185
        # X10 = self.residual_model_head1(x1)

        X01=self.residual_model_01(X00) + X00#---   208 ~ 211 ,AddBackward1213

        X10=self.downsample_model_10(X00) + self.residual_model_head2(x2)   #---   186 ~ 189
        X20=self.downsample_model_20(X10) + self.residual_model_head3(x3)  #---   190 ~ 193

        residual_11=self.residual_model_11(X10) + X10  #201 ~ 204    , sum  AddBackward1206
        downsample_11=self.downsample_model_11(X01)    #214 ~ 217
        X11=residual_11 + downsample_11      #---      AddBackward1218

        residual_21=self.residual_model_21(X20) + X20  #194 ~ 197  ,   sum  AddBackward1199
        downsample_21=self.downsample_model_21(X11)    #219 ~ 222
        X21=residual_21 + downsample_21                # AddBackward1223


        X24=self.residual_model_24(X21) + X21 #---   224 ~ 227 , AddBackward1229
        X25=self.residual_model_25(X24) + X24 #---   230 ~ 233 , AddBackward1235


        upsample_14=self.upsample_model_14(X24)       #242 ~ 246
        residual_14=self.residual_model_14(X11) + X11 #248 ~ 251, AddBackward1253
        X14=upsample_14 + residual_14   #---   AddBackward1254

        upsample_04=self.upsample_model_04(X14)       #268 ~ 272
        residual_04=self.residual_model_04(X01) + X01 #274 ~ 277, AddBackward1279
        X04=upsample_04 + residual_04   #---  AddBackward1280

        upsample_15=self.upsample_model_15(X25)       #236 ~ 240
        residual_15=self.residual_model_15(X14) + X14 #255 ~ 258, AddBackward1260
        X15=upsample_15 + residual_15   # AddBackward1261

        upsample_05=self.upsample_model_05(X15)   # 262 ~ 266
        residual_05=self.residual_model_05(X04) + X04  #281 ~ 284,AddBackward1286
        X05=upsample_05 + residual_05  # AddBackward1287

        X_tail=self.residual_model_tail(X05)    #288 ~ 291

        return X_tail


class FeatureExtractor(nn.Module):
    """The quadratic model"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.prelu1 = MyPReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.prelu2 = MyPReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.prelu3 = MyPReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.prelu4 = MyPReLU()
        self.conv5 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.prelu5 = MyPReLU()
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)
        self.prelu6 = MyPReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x1 = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x1))
        x2 = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x2))
        x3 = self.prelu6(self.conv6(x))

        return x1, x2, x3


def animeinterp_wrap(I1, I2, reuse_things, t):
    F12, Z1, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][1], reuse_things[4][2]
    F21, Z2, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][1], reuse_things[5][2]

    F1t = t * F12
    F2t = (1-t) * F21

    I1t = warp(I1, F1t, Z1, strMode='soft')
    one10 = torch.ones_like(I1, requires_grad=True)
    norm1 = warp(one10, F1t, Z1, strMode='soft')
    I1t[norm1 > 0] = I1t[norm1 > 0] / norm1[norm1 > 0]

    I2t = warp(I2, F2t, Z2, strMode='soft')
    one20 = torch.ones_like(I2, requires_grad=True)
    norm2 = warp(one20, F2t, Z2, strMode='soft')
    I2t[norm2 > 0] = I2t[norm2 > 0] / norm2[norm2 > 0]

    feat1t1 = warp(feat11, F1t, Z1, strMode='soft')
    one11 = torch.ones_like(feat11, requires_grad=True)
    norm1t1 = warp(one11, F1t, Z1, strMode='soft')
    feat1t1[norm1t1 > 0] = feat1t1[norm1t1 > 0] / norm1t1[norm1t1 > 0]

    feat2t1 = warp(feat21, F2t, Z2, strMode='soft')
    one21 = torch.ones_like(feat21, requires_grad=True)
    norm2t1 = warp(one21, F2t, Z2, strMode='soft')
    feat2t1[norm2t1 > 0] = feat2t1[norm2t1 > 0] / norm2t1[norm2t1 > 0]

    F1tdd = F.interpolate(F1t, scale_factor = 0.5, mode="bilinear") * 0.5
    Z1dd = F.interpolate(Z1, scale_factor = 0.5, mode="bilinear")
    feat1t2 = warp(feat12, F1tdd, Z1dd, strMode='soft')
    one12 = torch.ones_like(feat12, requires_grad=True)
    norm1t2 = warp(one12, F1tdd, Z1dd, strMode='soft')
    feat1t2[norm1t2 > 0] = feat1t2[norm1t2 > 0] / norm1t2[norm1t2 > 0]

    F2tdd = F.interpolate(F2t, scale_factor = 0.5, mode="bilinear") * 0.5
    Z2dd = F.interpolate(Z2, scale_factor = 0.5, mode="bilinear")
    feat2t2 = warp(feat22, F2tdd, Z2dd, strMode='soft')
    one22 = torch.ones_like(feat22, requires_grad=True)
    norm2t2 = warp(one22, F2tdd, Z2dd, strMode='soft')
    feat2t2[norm2t2 > 0] = feat2t2[norm2t2 > 0] / norm2t2[norm2t2 > 0]

    F1tddd = F.interpolate(F1t, scale_factor = 0.25, mode="bilinear") * 0.25
    Z1ddd = F.interpolate(Z1, scale_factor = 0.25, mode="bilinear")
    feat1t3 = warp(feat13, F1tddd, Z1ddd, strMode='soft')
    one13 = torch.ones_like(feat13, requires_grad=True)
    norm1t3 = warp(one13, F1tddd, Z1ddd, strMode='soft')
    feat1t3[norm1t3 > 0] = feat1t3[norm1t3 > 0] / norm1t3[norm1t3 > 0]

    F2tddd = F.interpolate(F2t, scale_factor = 0.25, mode="bilinear") * 0.25
    Z2ddd = F.interpolate(Z2, scale_factor = 0.25, mode="bilinear")
    feat2t3 = warp(feat23, F2tddd, Z2ddd, strMode='soft')
    one23 = torch.ones_like(feat23, requires_grad=True)
    norm2t3 = warp(one23, F2tddd, Z2ddd, strMode='soft')
    feat2t3[norm2t3 > 0] = feat2t3[norm2t3 > 0] / norm2t3[norm2t3 > 0]

    return I1t, I2t, feat1t1, feat2t1, feat1t2, feat2t2, feat1t3, feat2t3


torch.fx.wrap('animeinterp_wrap')


class AnimeInterp(nn.Module):
    """The quadratic model"""
    def __init__(self):
        super(AnimeInterp, self).__init__()
        self.feat_ext = FeatureExtractor()
        self.synnet = GridNet(6, 64, 128, 96*2, 3)

    def forward(self, I1, I2, reuse_things, merged, t):
        I1t, I2t, feat1t1, feat2t1, feat1t2, feat2t2, feat1t3, feat2t3 = animeinterp_wrap(I1, I2, reuse_things, t)
        It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1), merged)

        return torch.clamp(It_warp, 0, 1)
