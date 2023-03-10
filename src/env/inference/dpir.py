# -*- coding: utf-8 -*-
import os
import sys
import platform
import tempfile
import json
import vapoursynth as vs

from vapoursynth import core
from multiprocessing import cpu_count

ossystem = platform.system()

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
    streams = data['streams']
    model = data['model']
    strengthParam = data['strength']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()

cwd = os.getcwd()
vsmlrt_path = os.path.join(cwd, '..', 'python', 'Library', 'vstrt.dll')
core.std.LoadPlugin(path=vsmlrt_path)
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if model == 'Denoise':
    val = 255
if model == 'Deblock':
    val = 100

strength = strengthParam
noise_level = clip.std.BlankClip(format=vs.GRAYS, color=strength / val)

clip = core.trt.Model([clip, noise_level], engine_path=engine, num_streams=threading())

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()