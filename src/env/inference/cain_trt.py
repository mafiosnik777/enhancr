# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json

from multiprocessing import cpu_count

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.vfi_inference import vfi_frame_merger
from utils.trt_precision import check_model_precision_trt

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
    engine = data['engine']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    padding = data['padding']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    ToPadWidth = data['toPadWidth']
    ToPadHeight = data['toPadHeight']

core.num_threads = cpu_count() / 2

cwd = os.getcwd()
vsmlrt_path = os.path.join(cwd, '..', 'python', 'Library', 'vstrt.dll')
core.std.LoadPlugin(path=vsmlrt_path)

clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

if frameskip:
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    # use ssim for similarity calc
    clip = core.vmaf.Metric(clip, offs1, 2)

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.180)

if padding:
    clip = core.std.AddBorders(clip, right=ToPadWidth, top=ToPadHeight)

if check_model_precision_trt(engine) == "float32":
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
else:
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")
    print("Using fp16 i/o for inference", file=sys.stderr)

clip_pos1 = clip[1:]
clip_pos2 = clip.std.Trim(first=0,last=clip.num_frames-2)
clipstack =  [clip_pos1,clip_pos2]

output = core.trt.Model(
   clipstack,
   engine_path=engine,
   num_streams=int(streams)*4,
)
output=core.std.Interleave([clip,output])

clip1 = core.std.Interleave([clip, clip])

output = vfi_frame_merger(clip1, output)

# cain network specific padding
if padding:
    output = core.std.Crop(clip, right=ToPadWidth, top=ToPadHeight)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

output = vs.core.resize.Bicubic(output, format=vs.YUV422P8, matrix_s="709")

print("Starting video output | Threads: " + str(int(cpu_count() / 2)) + " | " + "Streams: " + streams, file=sys.stderr)
output.set_output()