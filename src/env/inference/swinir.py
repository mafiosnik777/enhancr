# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json
import math
import functools

import torch

from multiprocessing import cpu_count

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from arch.vs_swinir import swinir

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
    streams = data['streams']
    fp16 = data['fp16']
    tiling = data['tiling']
    frameskip = data['frameskip']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count() / 2

engine_path = os.path.join(os.getenv('APPDATA'), '.enhancr', 'models', 'engine')

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

if fp16:
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")
else:
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

def nvFuser():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    vram_gb = total_memory / (1024**3)
    if math.ceil(vram_gb) > 14:
        return True
    else:
        return False

if tiling:
    upscaled = swinir(clip=clip, model=1, device_index=0, num_streams=int(streams), tile_w=tileWidth, tile_h=tileHeight, cuda_graphs=True, nvfuser=nvFuser())
else:
    upscaled = swinir(clip=clip, model=1, device_index=0, num_streams=int(streams), cuda_graphs=True, nvfuser=nvFuser())

if frameskip:
    metric_thresh = 0.999
    partial = functools.partial(execute, upscaled=upscaled, metric_thresh=metric_thresh)
    clip = core.std.FrameEval(core.std.BlankClip(clip=upscaled, width=upscaled.width, height=upscaled.height), partial, prop_src=[clip])

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV422P8, matrix_s="709")

print("Starting video output | Threads: " + str(int(cpu_count() / 2)) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()