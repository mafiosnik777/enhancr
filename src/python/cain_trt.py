# -*- coding: utf-8 -*-
import os
import sys
import vapoursynth as vs
import platform
import tempfile
import json
import functools
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

# https://pyscenedetect.readthedocs.io/en/latest/reference/python-api/
def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    print("Finding Scene Changes.. (This could take a while on larger files)", file=sys.stderr)
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    result = scene_manager.get_scene_list()

    framelist = []
    from tqdm import tqdm

    for i in tqdm(range(len(result))):
        framelist.append(result[i][1].get_frames() * 2 - 1)
        # print(result[i][1].get_frames())
        # print(result[i][1].get_timecode())

    return framelist

# https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/57d2a6c32f61b681d0271ddbe72d2e7d18900d48/src/vfi_inference.py#L68
# ❤️ ty sudo u the goat
def vfi_frame_merger(
    clip1: vs.VideoNode,
    clip2: vs.VideoNode,
    skip_framelist=[],
) -> vs.VideoNode:
    core = vs.core

    def execute(n: int, clip1: vs.VideoNode, clip2: vs.VideoNode) -> vs.VideoNode:
        if (
            n in skip_framelist
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
with open(os.path.join(tmp_dir, "tmp.json"), encoding='utf-8') as f:
    data = json.load(f)
    video_path = data['file']
    frame_rate = data['framerate']
    engine = data['engine']
    streams = data['streams']
    sceneDetection = data['cain_scdetect']
    
clip = core.ffms2.Source(source=f"{video_path}", fpsnum=-1, fpsden=1, cache=False)

if sceneDetection == True:
    skip_frame_list = find_scenes(video_path, threshold=30)
else:
    skip_frame_list = []

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip1 = core.std.DeleteFrames(clip, frames=0)
clip2 = core.std.StackHorizontal([clip1, clip])

clip2 = core.trt.Model(
   clip2,
   engine_path=engine,
   num_streams=int(streams)*4,
)
clip2=core.std.Crop(clip2,right=clip.width)
clip1 = core.std.Interleave([clip, clip])
clip2 = core.std.Interleave([clip, clip2])

output = vfi_frame_merger(clip1, clip2, skip_frame_list)

output = vs.core.resize.Bicubic(output, format=vs.YUV420P8, matrix_s="709")

print("Starting video output..", file=sys.stderr)
output.set_output()