import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def vfi_inference(
    model_inference, clip: vs.VideoNode, multi=2, metric_thresh=0.999
) -> vs.VideoNode:
    core = vs.core

    if model_inference.amount_input_img == 4 and multi != 2:
        raise ValueError(
            "4 image input interpolation networks currently only support 2x"
        )

    def frame_to_tensor(frame: vs.VideoFrame):
        return np.stack(
            [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
        )

    def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
        for plane in range(f.format.num_planes):
            d = np.asarray(f[plane])
            np.copyto(d, array[plane, :, :])
        return f

    def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
        clip = core.std.BlankClip(
            clip=clip, width=image.shape[-1], height=image.shape[-2]
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: tensor_to_frame(f.copy(), image),
        )

    def execute(n: int, clip0: vs.VideoNode, clip1: vs.VideoNode) -> vs.VideoNode:
        clip_metric = clip0.get_frame(n).props.get("float_ssim")

        if (
            (n % multi == 0)
            or n == 0
            or clip0.get_frame(n).props.get("_SceneChangeNext")
            or (clip_metric and clip_metric > metric_thresh)
            or n // multi == clip.num_frames - 1
        ):
            return clip0

        I0 = frame_to_tensor(clip0.get_frame(n))
        I1 = frame_to_tensor(clip1.get_frame(n))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

        # clamping because vs does not give tensors in range 0-1, results in nan in output
        I0 = torch.clamp(I0, min=0, max=1)
        I1 = torch.clamp(I1, min=0, max=1)

        with torch.inference_mode():
            middle = model_inference.execute(I0, I1, (n % multi) / multi)

        return tensor_to_clip(clip=clip0, image=middle)

    def execute_4img(n: int, clip0: vs.VideoNode, clip1: vs.VideoNode) -> vs.VideoNode:
        clip_metric = clip0.get_frame(n).props.get("float_ssim")

        if (
            (n % multi == 0)
            or n == 0
            or n == 1
            or clip0.get_frame(n).props.get("_SceneChangeNext")
            or (clip_metric and clip_metric > metric_thresh)
            or n // multi == clip.num_frames - 1
        ):
            return clip0

        I0 = frame_to_tensor(clip0.get_frame(n - 1 - 1))
        I1 = frame_to_tensor(clip1.get_frame(n - 1 - 1))
        I2 = frame_to_tensor(clip0.get_frame(n + 3 - 1))
        I3 = frame_to_tensor(clip1.get_frame(n + 3 - 1))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)
        I2 = torch.Tensor(I2).unsqueeze(0).to("cuda", non_blocking=True)
        I3 = torch.Tensor(I3).unsqueeze(0).to("cuda", non_blocking=True)

        # clamping because vs does not give tensors in range 0-1, results in nan in output
        I0 = torch.clamp(I0, min=0, max=1)
        I1 = torch.clamp(I1, min=0, max=1)
        I2 = torch.clamp(I2, min=0, max=1)
        I3 = torch.clamp(I3, min=0, max=1)

        with torch.inference_mode():
            middle = model_inference.execute(I0, I1, I2, I3)

        return tensor_to_clip(clip=clip0, image=middle)

    cache = {}

    def execute_cache(n: int, clip0: vs.VideoNode, clip1: vs.VideoNode) -> vs.VideoNode:
        clip_metric = clip0.get_frame(n).props.get("float_ssim")

        if (
            (n % multi == 0)
            or n == 0
            or clip0.get_frame(n).props.get("_SceneChangeNext")
            or (clip_metric and clip_metric > metric_thresh)
            or n // multi == clip.num_frames - 1
        ):
            return clip0

        if str(n) not in cache:
            cache.clear()

            I0 = frame_to_tensor(clip0.get_frame(n))
            I1 = frame_to_tensor(clip1.get_frame(n))

            I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
            I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

            # clamping because vs does not give tensors in range 0-1, results in nan in output
            I0 = torch.clamp(I0, min=0, max=1)
            I1 = torch.clamp(I1, min=0, max=1)

            output = model_inference.execute(I0, I1, multi)

            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i, :, :, :].cpu().numpy()

            del output
            torch.cuda.empty_cache()
        return tensor_to_clip(clip=clip0, image=cache[str(n)])

    clip0 = vs.core.std.Interleave([clip] * multi)
    clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(
        frames=0
    )
    clip1 = vs.core.std.Interleave([clip1] * multi)

    if model_inference.amount_input_img == 4:
        return core.std.FrameEval(
            core.std.BlankClip(clip=clip0, width=clip.width, height=clip.height),
            functools.partial(execute_4img, clip0=clip0, clip1=clip1),
        )

    if model_inference.cache:
        return core.std.FrameEval(
            core.std.BlankClip(clip=clip0, width=clip.width, height=clip.height),
            functools.partial(execute_cache, clip0=clip0, clip1=clip1),
        )
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip0, width=clip.width, height=clip.height),
        functools.partial(execute, clip0=clip0, clip1=clip1),
    )


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