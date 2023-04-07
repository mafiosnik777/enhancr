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

from arch.gmfupss_torch.GMFUpSS import GMFupSS
from arch.gmfss_union_torch.GMFSS_Union import GMFSS_Union
from arch.gmfss_fortuna_torch.GMFSS_Fortuna import GMFSS_Fortuna
from arch.gmfss_fortuna_union_torch.GMFSS_Fortuna_Union import GMFSS_Fortuna_union
from utils.vfi_inference import vfi_inference

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
    model = data['model']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    padding = data['padding']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    ToPadWidth = data['toPadWidth']
    ToPadHeight = data['toPadHeight']
    precision = data['halfPrecision']

clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()

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

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if model == "GMFSS - Up":
    model_inference = GMFupSS(partial_fp16=precision)
elif model == "GMFSS - Union":
    model_inference = GMFSS_Union(partial_fp16=precision)
elif model == "GMFSS - Fortuna":
    model_inference = GMFSS_Fortuna(partial_fp16=precision)
elif model == "GMFSS - Fortuna - Union":
    model_inference = GMFSS_Fortuna_union(partial_fp16=precision)

clip = vfi_inference(
       model_inference=model_inference, clip=clip, multi=2, metric_thresh=0.999
)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV422P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(1), file=sys.stderr)
clip.set_output()