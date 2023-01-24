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
    rife_model = data['model']
    TTA = data['rife_tta']
    UHD = data['rife_uhd']
    sceneDetection = data['sc']
    frameskip = data['skip']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    
# get rife model
if rife_model == 'RIFE - v2.3':
    model = 5
elif rife_model == 'RIFE - v4.0' and TTA == True:
    model = 10
    TTA = False
elif rife_model == 'RIFE - v4.0':
    model = 9
elif rife_model == 'RIFE - v4.5' and TTA == True:
    model = 20
    TTA = False
elif rife_model == 'RIFE - v4.5':
    model = 19
elif rife_model == 'RIFE - v4.6' and TTA == True:
    model = 22
    TTA = False
elif rife_model == 'RIFE - v4.6':
    model = 21
    TTA = False
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.180)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = core.rife.RIFE(clip, model=model, factor_num=2, gpu_id=0, gpu_thread=2, tta=TTA, uhd=UHD, skip=frameskip, sc=sceneDetection)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
clip.set_output()
