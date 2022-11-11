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

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp_dir, "tmp.json"), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    engine = data['engine']
    streams = data['streams']
    tiling = data['tiling']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])

    
clip = core.ffms2.Source(source=f"{video_path}", fpsnum=-1, fpsden=1, cache=False)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if tiling == False:
    clip = core.trt.Model(clip, engine_path=engine, num_streams=streams)
else:
    clip = core.trt.Model(clip, engine_path=engine, num_streams=streams, tilesize=[tileHeight, tileWidth])

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()
