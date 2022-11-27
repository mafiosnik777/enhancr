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

# https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/1c152506f5c76b76708524398483c6c97d42cd95/src/rife_trt.py#L6
# ❤️ ty sudo u the goat
def vfi_frame_merger(
    clip1: vs.VideoNode,
    clip2: vs.VideoNode,
) -> vs.VideoNode:
    core = vs.core

    metric_thresh = 0.999

    def execute(n: int, clip1: vs.VideoNode, clip2: vs.VideoNode) -> vs.VideoNode:
        ssim_clip = clip1.get_frame(n).props.get("float_ssim")
        if (ssim_clip and ssim_clip > metric_thresh) or clip1.get_frame(n).props.get(
            "_SceneChangeNext"
        ):
            return clip1
        return clip2

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip1, width=clip1.width, height=clip1.height),
        functools.partial(execute, clip1=clip1, clip2=clip2),
    ) 

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
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 4

if ossystem == "Windows":
    tmp_dir = tempfile.gettempdir() + "\\enhancr\\"
else:
    tmp_dir = tempfile.gettempdir() + "/enhancr/"

# load json with input file path and framerate
with open(os.path.join(tmp_dir, "tmp.json"), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    frame_rate = data['framerate']
    engine = data['rife_engine']
    streams = data['streams']
    sceneDetection = data['rife_scdetect']
    
clip = core.ffms2.Source(source=f"{video_path}", fpsnum=-1, fpsden=1, cache=False)

if sceneDetection == True:
    clip = core.misc.SCDetect(clip=clip, threshold=0.100)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = rife_trt(clip, multi = 2, scale = 1.0, device_id = 0, num_streams = streams, engine_path = engine)

output = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
output.set_output()