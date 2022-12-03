# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
from decimal import Decimal
from fractions import Fraction
import tempfile
import json
ossystem = platform.system()

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 4

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    frame_rate = data['framerate']
    sceneDetection = data['sc']
    model = data['model']
    frameskip = data['skip']

if model == 'RVP - v1.0':
    cainModel = 0
if model == 'CVP - v6.0':
    cainModel = 1
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

sceneDetectionThreshold = 0.200
clip = core.misc.SCDetect(clip=clip, threshold=sceneDetectionThreshold)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV444PS, matrix_in_s="709")

clip = core.cain.CAIN(clip, model=cainModel, gpu_id=0, gpu_thread=2, sc=sceneDetection, skip=frameskip)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
print("Starting video output..", file=sys.stderr)
clip.set_output()
