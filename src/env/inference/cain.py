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
    sceneDetection = data['sc']
    model = data['model']
    frameskip = data['skip']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    streams = data['streams']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()

if model == 'RVP - v1.0':
    cainModel = 0
if model == 'CVP - v6.0':
    cainModel = 1
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.180)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV444PS, matrix_in_s="709")

clip = core.cain.CAIN(clip, model=cainModel, gpu_id=0, gpu_thread=threading(), sc=sceneDetection, skip=frameskip)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()
