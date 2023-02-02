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
    engine = data['engine']
    streams = data['streams']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()
    
clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = core.w2xnvk.Waifu2x(clip, noise=2, scale=2, model=0, precision=16, gpu_thread=threading())

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(threading()), file=sys.stderr)
clip.set_output()
