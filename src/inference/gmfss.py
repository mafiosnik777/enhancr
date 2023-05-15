# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json

from multiprocessing import cpu_count

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from arch.vsgmfss_union import gmfss_union
from arch.vsgmfss_fortuna import gmfss_fortuna

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
    frame_rate = data['framerate']
    engine = data['engine']
    model = data['model']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    padding = data['padding']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    ToPadWidth = data['toPadWidth']
    ToPadHeight = data['toPadHeight']
    halfPrecision = data['halfPrecision']

clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count() / 2

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.180)

if halfPrecision:
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")
    print("Using fp16 i/o for inference", file=sys.stderr)
else:
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if model == "GMFSS - Union":
    clip = gmfss_union(clip, num_streams=threading(), trt=False, model=0)
elif model == "GMFSS - Fortuna":
    clip = gmfss_fortuna(clip, num_streams=threading(), trt=False, model=0)
elif model == "GMFSS - Fortuna - Union":
    clip = gmfss_fortuna(clip, num_streams=threading(), trt=False, model=1)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV422P8, matrix_s="709")

print("Starting video output | Threads: " + str(int(cpu_count() / 2)) + " | " + "Streams: " + str(1), file=sys.stderr)
clip.set_output()