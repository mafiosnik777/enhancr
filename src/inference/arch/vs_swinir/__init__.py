from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functorch.compile import memory_efficient_fusion

from .network_swinir import SwinIR

__version__ = "2.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

package_dir = os.path.dirname(os.path.realpath(__file__))


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class CUDAGraphs:
        graph: list[torch.cuda.CUDAGraph]
        static_input: list[torch.Tensor]
        static_output: list[torch.Tensor]


@torch.inference_mode()
def swinir(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    nvfuser: bool = False,
    cuda_graphs: bool = False,
    model: int = 0,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 16,
) -> vs.VideoNode:
    """Image Restoration Using Swin Transformer

    :param clip:            Clip to process. Only RGBH and RGBS formats are supported.
                            RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:    Device ordinal of the GPU.
    :param num_streams:     Number of CUDA streams to enqueue the kernels.
    :param nvfuser:         Enable fusion through nvFuser. (experimental)
    :param cuda_graphs:     Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels sequentially.
    :param model:           Model to use.
                            0 = lightweightSR_DIV2K_s64w8_SwinIR-S_x2
                            1 = lightweightSR_DIV2K_s64w8_SwinIR-S_x3
                            2 = lightweightSR_DIV2K_s64w8_SwinIR-S_x4
                            3 = realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN
                            4 = 2x_Bubble_AnimeScale_SwinIR_Small_v1
    :param tile_w:          Tile width. As too large images result in the out of GPU memory issue, so this tile option
                            will first crop input images into tiles, and then process each of them. Finally, they will
                            be merged into one image. 0 denotes for do not use tile.
    :param tile_h:          Tile height.
    :param tile_pad:        Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("swinir: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("swinir: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("swinir: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("swinir: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("swinir: setting num_streams greater than `core.num_threads` is useless")

    if model not in range(5):
        raise vs.Error("swinir: model must be 0, 1, 2, 3, or 4")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case 1:
            model_name = "2x_Bubble_AnimeScale_SwinIR_Small_v1.pth"
            module = SwinIR(
                upscale=2,
                img_size=32,
                window_size=8,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffle",
                resi_connection="1conv",
            )
            scale = 2

    model_path = os.path.join(package_dir, model_name)

    pretrained_model = torch.load(model_path, map_location="cpu")
    if "params_ema" in pretrained_model:
        pretrained_model = pretrained_model["params_ema"]
    elif "params" in pretrained_model:
        pretrained_model = pretrained_model["params"]

    module.load_state_dict(pretrained_model)
    module.eval().to(device, memory_format=torch.channels_last)
    if fp16:
        module.half()

    if tile_w > 0 and tile_h > 0:
        pad_w = math.ceil(min(tile_w + 2 * tile_pad, clip.width) / 8) * 8
        pad_h = math.ceil(min(tile_h + 2 * tile_pad, clip.height) / 8) * 8
    else:
        pad_w = math.ceil(clip.width / 8) * 8
        pad_h = math.ceil(clip.height / 8) * 8

    if nvfuser:
        module = memory_efficient_fusion(module)

    if cuda_graphs:
        graph: list[torch.cuda.CUDAGraph] = []
        static_input: list[torch.Tensor] = []
        static_output: list[torch.Tensor] = []

        for i in range(num_streams):
            static_input.append(
                torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device).to(memory_format=torch.channels_last)
            )

            torch.cuda.synchronize(device=device)
            stream[i].wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream[i]):
                module(static_input[i])
            torch.cuda.current_stream(device=device).wait_stream(stream[i])
            torch.cuda.synchronize(device=device)

            graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graph[i], stream=stream[i]):
                static_output.append(module(static_input[i]))

        backend = Backend.CUDAGraphs(graph, static_input, static_output)
    else:
        backend = Backend.Eager(module)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "reflect")

                if cuda_graphs:
                    static_input[local_index].copy_(img)
                    graph[local_index].replay()
                    output = static_output[local_index]
                else:
                    output = module(img)

                output = output[:, :, : h * scale, : w * scale]

            return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.CUDAGraphs,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), mode)

            # process tile
            if isinstance(backend, Backend.CUDAGraphs):
                backend.static_input[index].copy_(input_tile)
                backend.graph[index].replay()
                output_tile = backend.static_output[index]
            else:
                output_tile = backend.module(input_tile)

            output_tile = output_tile[:, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
