# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json
import functools

#https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/f0a30ebcae0cd9b50a07aadf152a23f74f9ba187/src/vfi_inference.py#L159
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
    engine = data['engine']
    streams = data['streams']
    sceneDetection = data['sc']
    frameskip = data['skip']
    padding = data['padding']
    sensitivity = data['sensitivity']
    sensitivityValue = data['sensitivityValue']
    ToPadWidth = data['toPadWidth']
    ToPadHeight = data['toPadHeight']

clip = core.lsmas.LWLibavSource(source=f"{video_path}", cache=0)

if frameskip:
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    # use ssim for similarity calc
    clip = core.vmaf.Metric(clip, offs1, 2)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

if sceneDetection:
    if sensitivity:
        clip = core.misc.SCDetect(clip=clip, threshold=sensitivityValue)
    else:
        clip = core.misc.SCDetect(clip=clip, threshold=0.100)

if padding:
    clip = core.std.AddBorders(clip, right=ToPadWidth, top=ToPadHeight)

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

if padding:
    output = core.std.Crop(clip, right=ToPadWidth, top=ToPadHeight)

output = vs.core.resize.Bicubic(output, format=vs.YUV422P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
output.set_output()