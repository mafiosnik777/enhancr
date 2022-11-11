# -*- coding: utf-8 -*-
from decimal import Decimal
import os
import sys
from decimal import Decimal
from fractions import Fraction
import vapoursynth as vs
import platform
import tempfile
import json
ossystem = platform.system()

core = vs.core

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
    
clip = core.ffms2.Source(source=f"{video_path}", fpsnum=-1, fpsden=1, cache=False)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = core.w2xnvk.Waifu2x(clip, noise=2, scale=2, model=0, precision=16)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()
