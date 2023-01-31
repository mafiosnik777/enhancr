# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from arch.vsgmfss_union import gmfss_union
from utils.vfi_inference import vfi_frame_merger

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
    precision = data['halfPrecision']

core.num_threads = 4

engine_path = os.path.join(os.getenv('APPDATA'), '.enhancr', 'models', 'engine')

clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

if frameskip:
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    # use ssim for similarity calc
    clip = core.vmaf.Metric(clip, offs1, 2)

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.180)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

clip = gmfss_union(clip, num_streams=int(streams), trt=True, trt_cache_path=engine_path, model=0)

clip1 = core.std.Interleave([clip, clip])
output = vfi_frame_merger(clip1, clip)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV422P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()