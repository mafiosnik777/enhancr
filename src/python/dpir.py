# -*- coding: utf-8 -*-
from asyncio import streams
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json

import vapoursynth as vs
from vapoursynth import core

ossystem = platform.system()

vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 8

# already loaded in python
# if ossystem == "Windows":
#     core.std.LoadPlugin(path="vsynth/vapoursynth64/plugins/ffms2.dll")

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp_dir, "tmp.json"), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    frame_rate = data['framerate']
    engine = data['engine']
    streams = data['streams']
    model = data['model']
    strengthParam = data['strength']
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if model == 'Denoise':
    val = 255
if model == 'Deblock':
    val = 100

strength = strengthParam
noise_level = clip.std.BlankClip(format=vs.GRAYS, color=strength / val)

clip = core.trt.Model([clip, noise_level], engine_path=engine, num_streams=streams)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()