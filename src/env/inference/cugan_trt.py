# -*- coding: utf-8 -*-
import os
import sys
import platform
import tempfile
import json
import vapoursynth as vs
import functools

from vapoursynth import core
from multiprocessing import cpu_count

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.trt_precision import check_model_precision_trt

ossystem = platform.system()

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    engine = data['engine']
    streams = data['streams']
    tiling = data['tiling']
    frameskip = data['frameskip']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count() / 2

cwd = os.getcwd()
vsmlrt_path = os.path.join(cwd, '..', 'python', 'Library', 'vstrt.dll')
core.std.LoadPlugin(path=vsmlrt_path)

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

if check_model_precision_trt(engine) == "float32":
    clip = vs.core.resize.Spline64(clip, format=vs.RGBS, matrix_in_s="709", transfer_in_s="linear")
else:
    clip = vs.core.resize.Spline64(clip, format=vs.RGBH, matrix_in_s="709", transfer_in_s="linear")
    print("Using fp16 i/o for inference", file=sys.stderr)

if tiling == False:
    upscaled = core.trt.Model(clip, engine_path=engine, num_streams=threading())
else:
    upscaled = core.trt.Model(clip, engine_path=engine, num_streams=threading(), tilesize=[tileHeight, tileWidth])

if frameskip:
    metric_thresh = 0.999
    partial = functools.partial(execute, upscaled=upscaled, metric_thresh=metric_thresh)
    clip = core.std.FrameEval(core.std.BlankClip(clip=upscaled, width=upscaled.width, height=upscaled.height), partial, prop_src=[clip])

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(int(cpu_count() / 2)) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()
