# -*- coding: utf-8 -*-
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

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    engine = data['engine']
    tiling = data['tiling']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])
    fp16 = data['fp16']

    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if tiling == False:
    clip = core.ncnn.Model(clip, network_path=engine, num_streams=1, fp16=fp16)
else:
    clip = core.ncnn.Model(clip, network_path=engine, num_streams=1, fp16=fp16, tilesize=[tileHeight, tileWidth])

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()