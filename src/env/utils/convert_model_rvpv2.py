import argparse
import sys

import numpy as np

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="rvpv2 onnx conversion script")
parser.add_argument("--input", metavar="--input", type=str, help="input model")
parser.add_argument("--output", metavar="--output", type=str, help="output model")
parser.add_argument("--tmp", metavar="--tmp", type=str, help="temp folder")
parser.add_argument("--width", metavar="--width", type=int, help="width")
parser.add_argument("--height", metavar="--height", type=int, help="height")
parser.add_argument("--fp16", metavar="--fp16", type=bool, help="fp16 precision")
args = parser.parse_args()

def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean


from arch.dynamicconv import *

nof_kernels_param = 4
reduce_param = 4


def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        #default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


# https://github.com/fangwei123456/PixelUnshuffle-pytorch/blob/master/PixelUnshuffle/__init__.py
def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1

    # GatedConv2dWithActivation, doconv, TBC and dynamic does not support kernel as input, using normal conv2d because of this
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # because of tensorrt
        self.reflection_pad = torch.nn.ZeroPad2d(reflection_padding)

        self.conv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=in_feat, out_channels=out_feat, stride=stride, kernel_size=kernel_size, bias=True)


    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out





class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter =  DynamicConvolution(nof_kernels_param, reduce_param, in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)   

            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter =  DynamicConvolution(nof_kernels_param, reduce_param, in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter =  DynamicConvolution(nof_kernels_param, reduce_param, in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0)  

            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=False, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res

        return out 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight

        self.conv_du = nn.Sequential(
            DynamicConvolution(nof_kernels_param, reduce_param, in_channels=channel, out_channels= (channel // reduction), kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            DynamicConvolution(nof_kernels_param, reduce_param, in_channels=(channel // reduction), out_channels=channel, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=False,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res
        return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()
        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=False, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()

        self.headConv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=n_feats*2, out_channels=n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)

        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=12,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(2)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=n_feats, out_channels=n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3) 

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        self.shuffler = torch.nn.PixelUnshuffle(2**depth)
        # custom unshuffle
        #self.shuffler = PixelUnshuffle(2**depth)
        relu = nn.LeakyReLU(0.2, True)
        self.interpolate = Interpolation(5, 12, in_channels * (4**depth), act=relu)

        self.frdb11 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb12 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb13 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb14 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)
        self.frdb15 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)

        self.frdb21 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb22 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb23 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.frdb24 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)
        self.frdb25 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)

        # after interpolate
        self.rdb = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb2 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb3 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb4 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb5 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb6 = ResidualDenseBlock(num_feat=192, num_grow_ch=32)
        self.rdb7 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)
        self.rdb8 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)
        self.rdb9 = ResidualDenseBlock(num_feat=192, num_grow_ch=64)

    def forward(self, x1, x2):
        feats1 = self.shuffler(x1) # torch.Size([1, 192, 90, 160])
        feats2 = self.shuffler(x2)

        feats1 = self.frdb11(feats1)
        feats1 = self.frdb12(feats1)
        feats1 = self.frdb13(feats1)
        feats1 = self.frdb14(feats1)
        feats1 = self.frdb15(feats1)

        feats2 = self.frdb21(feats2)
        feats2 = self.frdb22(feats2)
        feats2 = self.frdb23(feats2)
        feats2 = self.frdb24(feats2)
        feats2 = self.frdb25(feats2)

        feats = self.interpolate(feats1, feats2)
        # torch.Size([1, 192, 16, 16])
        #print(feats.shape)

        #feats = self.conv_first(feats)
        feats = self.rdb(feats)
        feats = self.rdb2(feats)
        feats = self.rdb3(feats)        
        feats = self.rdb4(feats)
        feats = self.rdb5(feats)
        feats = self.rdb6(feats)
        feats = self.rdb7(feats)
        feats = self.rdb8(feats)
        feats = self.rdb9(feats)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()
        self.shuffler = torch.nn.PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3):
        super(CAIN, self).__init__()
        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x2):
        x2, x1 = torch.split(x2, 3, dim=1)
        width=x1.shape[3]
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)
        out = self.decoder(self.encoder(x1, x2))
        mi = (m1 + m2) / 2
        out += mi
        out=out[:,:,:,0:width]
        return out


model = CAIN(3)
model.load_state_dict(torch.load(args.input, map_location=torch.device('cuda')), strict=False)

if args.fp16:
    device = torch.device("cuda")
    model.half()
    model = model.to(device)

input_names = ["input"]
output_names = ["output"]

if args.fp16:
    f1 = torch.rand((1, 6, args.height, args.width)).half().to(device)
else:
    f1 = torch.rand((1, 6, args.height, args.width))
    
x = f1

import os

torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    os.path.join(args.tmp, 'rvpv2_tmp.onnx'),  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=17,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=input_names,  # the model's input names
    output_names=output_names)#
del model
print("[Conversion] Successfully converted model to onnx: " + "'" + args.input + "'", file=sys.stderr)

import onnx
from onnxsim import simplify

tempOnnx = onnx.load(os.path.join(args.tmp, 'rvpv2_tmp.onnx'))

print("[Conversion] Simplifying model... ", file=sys.stderr)
simplified_model, check = simplify(tempOnnx)

assert check, "[Error] Simplified ONNX model could not be validated"
onnx.checker.check_model(simplified_model)

onnx.save(simplified_model, os.path.join(args.output))
print("[Conversion] Successfully simplified onnx model: " + "'" + os.path.join(args.tmp, 'rvpv2_tmp.onnx') + "'", file=sys.stderr)

