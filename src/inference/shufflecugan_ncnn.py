# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json
import vapoursynth as vs
import functools

from multiprocessing import cpu_count

ossystem = platform.system()
core = vs.core

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    engine = data['onnx']
    tiling = data['tiling']
    frameskip = data['frameskip']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])
    fp16 = data['fp16']
    streams = data['streams']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count() / 2
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

def execute(n, upscaled, metric_thresh, f):
    ssim_clip = f.props.get("float_ssim")

    if n == 0 or n == len(upscaled) - 1 or (ssim_clip and ssim_clip > metric_thresh):
        return upscaled
    else:
        return upscaled[n-1]

offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
offs1 = core.std.CopyFrameProps(offs1, clip)
clip = core.vmaf.Metric(clip, offs1, 2)

clip = vs.core.resize.Spline64(clip, format=vs.RGBS, matrix_in_s="709", transfer_in_s="linear")

# ncnn related padding
def pad(n):
    return ((n + 7) // 8) * 8 - n

paddedHeight = False
paddedWidth = False
pxw = pad(clip.width)
pxh = pad(clip.height)

if (clip.height % 8 != 0):
    clip = core.std.AddBorders(clip, bottom=pad(clip.height))
    paddedHeight = True
    
if (clip.width % 8 != 0):
    clip = core.std.AddBorders(clip, right=pad(clip.width))
    paddedWidth = True

in_tile_channels = 3
in_tile_height = clip.height
in_tile_width = clip.width
out_tile_channels = 3
out_tile_height = clip.height * 2
out_tile_width = clip.width * 2

ncnn_shape_hint = (in_tile_channels, in_tile_height, in_tile_width, out_tile_channels, out_tile_height, out_tile_width)

if tiling == True:
    upscaled = core.ncnn.Model(clip, network_path=engine, num_streams=threading(), fp16=fp16, use_ncnn_network_format=True, ncnn_shape_hint=ncnn_shape_hint)
else:
    upscaled = core.ncnn.Model(clip, network_path=engine, num_streams=threading(), fp16=fp16, use_ncnn_network_format=True, ncnn_shape_hint=ncnn_shape_hint, tilesize=[tileHeight, tileWidth])

if frameskip:
    metric_thresh = 0.999
    partial = functools.partial(execute, upscaled=upscaled, metric_thresh=metric_thresh)
    upscaled = core.std.FrameEval(core.std.BlankClip(clip=upscaled, width=upscaled.width, height=upscaled.height), partial, prop_src=[clip])

clip = vs.core.resize.Bicubic(upscaled, format=vs.YUV420P8, matrix_s="709")

if paddedHeight:
    clip = core.std.Crop(clip, bottom=pxh*2)

if paddedWidth:
    clip = core.std.Crop(clip, right=pxw*2)

print("Starting video output | Threads: " + str(int(cpu_count() / 2)) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()