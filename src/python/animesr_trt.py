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
with open(os.path.join(tmp_dir, "tmp.json"), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    engine = data['engine']
    streams = data['streams']
    tiling = data['tiling']
    tileHeight = int(data['tileHeight'])
    tileWidth = int(data['tileWidth'])

    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip_pos1 = clip[1:]
clip_pos2 = clip.std.Trim(first=0,last=clip.num_frames-2)
clipstack =  [clip_pos1,clip_pos2]

if tiling == False:
    clip = core.trt.Model(clipstack, engine_path=engine, num_streams=streams)
else:
    clip = core.trt.Model(clipstack, engine_path=engine, num_streams=streams, tilesize=[tileHeight, tileWidth])

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()