# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
from vapoursynth import core
import platform
import tempfile
import json
import functools
import math

from multiprocessing import cpu_count

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.vfi_inference import vfi_frame_merger

def rife_trt(
    clip: vs.VideoNode,
    multi: int = 2,
    scale: float = 1.0,
    device_id: int = 0,
    num_streams: int = 1,
    engine_path: str = "",
):

    initial = core.std.Interleave([clip] * (multi - 1))

    terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    terminal = core.std.Interleave([terminal] * (multi - 1))

    timepoint = core.std.Interleave(
        [
            clip.std.BlankClip(format=vs.GRAYS, color=i / multi, length=1)
            for i in range(1, multi)
        ]
    ).std.Loop(clip.num_frames)

    scale = core.std.Interleave(
        [
            clip.std.BlankClip(format=vs.GRAYS, color=scale, length=1)
            for i in range(1, multi)
        ]
    ).std.Loop(clip.num_frames)

    clips = [initial, terminal, timepoint, scale]

    output = core.trt.Model(
        clips, engine_path, device_id=device_id, num_streams=num_streams
    )

    # using FrameEval is much faster than calling core.akarin.Expr
    output = vfi_frame_merger(initial, output)

    if multi == 2:
        return core.std.Interleave([clip, output])
    else:
        return core.std.Interleave(
            [
                clip,
                *(
                    output.std.SelectEvery(cycle=multi - 1, offsets=i)
                    for i in range(multi - 1)
                ),
            ]
        )

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
    engine = data['rife_engine']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']

def threading():
  return int(streams) if int(streams) < cpu_count() else cpu_count()
core.num_threads = cpu_count()

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

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = rife_trt(clip, multi = 2, scale = 1.0, device_id = 0, num_streams = threading(), engine_path = engine)

# padding if clip dimensions aren't divisble by 2
if (clip.height % 2 != 0):
    clip = core.std.AddBorders(clip, bottom=1)
    
if (clip.width % 2 != 0):
    clip = core.std.AddBorders(clip, right=1)

output = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output | Threads: " + str(cpu_count()) + " | " + "Streams: " + str(threading()), file=sys.stderr)
output.set_output()