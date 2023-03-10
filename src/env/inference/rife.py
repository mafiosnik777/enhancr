# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json

from multiprocessing import cpu_count

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
    rife_model = data['model']
    TTA = data['rife_tta']
    UHD = data['rife_uhd']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()
    
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

clip = core.rife.RIFE(clip, model=model, factor_num=2, gpu_id=0, gpu_thread=threading(), tta=TTA, uhd=UHD, skip=frameskip, sc=sceneDetection)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()
