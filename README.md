# ![heading-icon](https://mafiosnik.needsmental.help/4pBrFtuv8T.png?key=UheGMMXIDOTdJF)

**enhancr** is an **elegant and easy to use** GUI for **Video Frame Interpolation** and **Video Upscaling** which takes advantage of artificial intelligence - built using **node.js** and **Electron**. It was created **to enhance the user experience** for anyone interested in enhancing video footage using artificial intelligence. The GUI was **designed to provide a stunning experience** powered by state-of-the-art technologies **without feeling clunky and outdated** like other alternatives.

![gui-preview-image](https://mafiosnik.needsmental.help/1wLGH7CxEA.png?key=uk7M7TaolACipq)

It features blazing-fast **TensorRT** inference by NVIDIA, which can speed up AI processes **significantly**. **Pre-packaged, without the need to install Docker or WSL** (Windows Subsystem for Linux) - and **NCNN** inference by Tencent which is lightweight and runs on **NVIDIA**, **AMD** and even **Apple Silicon** - in contrast to the mammoth of an inference PyTorch is, which **only runs on NVIDA GPUs**.

# Features

## Interpolation

>**RIFE (NCNN)** - [megvii-research](https://github.com/megvii-research)/**[ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)** - powered by [styler00dollar](https://github.com/styler00dollar)/**[VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan)**

>**RIFE (TensorRT)** - [megvii-research](https://github.com/megvii-research)/**[ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

>**CAIN (NCNN)** - [myungsub](https://github.com/myungsub)/**[CAIN](https://github.com/myungsub/CAIN)** - powered by [mafiosnik](https://github.com/mafiosnik777)/**vsynth-cain-ncnn-vulkan** (unreleased)

>**CAIN (TensorRT)** - [myungsub](https://github.com/myungsub)/**[CAIN](https://github.com/myungsub/CAIN)** - powered by [HubertSotnowski](https://github.com/HubertSotnowski)/**vsynth-cain-tensorrt** (unreleased)

*Thanks to [HubertSontowski](https://github.com/HubertSotnowski) and [styler00dollar](https://github.com/styler00dollar) for helping out with implementing CAIN.*

## Upscaling

>**waifu2x (NCNN)** - [nagadomi](https://github.com/nagadomi)/**[waifu2x](https://github.com/nagadomi/waifu2x)** - powered by [nlzy](https://github.com/nlzy)/**[vs-mlrt](https://github.com/nlzy/vapoursynth-waifu2x-ncnn-vulkan)**

>**RealESRGAN (TensorRT)** - [xinntao](https://github.com/xinntao)/**[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

>**RealCUGAN (ONNX)** - [bilibili](https://github.com/bilibili)/[ailab](https://github.com/bilibili/ailab)/**[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

## Restoration

>**DPIR (TensorRT)** - [cszn](https://github.com/cszn)/**[DPIR](https://github.com/cszn/DPIR)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

# System Requirements

#### Minimum:
 - Dual Core CPU with Hyperthreading enabled
 - Vulkan-capable graphics processor
 - Windows 10

#### Recommended:

-   Quad Core Intel Kaby Lake/AMD Ryzen or newer with Hyperthreading enabled
-   16 GB RAM
-   NVIDIA 1000 Series (Pascal) for TensorRT/NCNN or AMD Polaris for NCNN
-   Windows 11

# Installation


Download the installer on [**Patreon**](https://www.patreon.com/mafiosnik) for **newest** pre-packaged builds with **all dependencies**. 
This repository **does not** provide builds & the embeddable python environment that's included on Patreon.

![installer](https://mafiosnik.needsmental.help/mEerVMP8LA.png?key=bzdnzy2RYJGOvO)

>*There probably will be free versions down the line, after a backlog of versions has released.*

# macOS and Linux Support

The GUI was built with cross-platform in mind and references both operating systems already.
**The current focus lies on delivering a stable & working solution for Windows first, but support for Linux and macOS will follow very soon.**

![enhancr-macos](https://mafiosnik.needsmental.help/st6TAP6g9t.png?key=kfq89rqeM2kEdi)

Support for Apple Silicon is planned as well, but I currently only have an Intel Macbook Pro available for testing.

# Benchmarks

Input size: 1920x1080 @ 2x

|| RTX 2060 Super | RTX 4090
|--|--|--|
| RIFE / rife-v4.6 (NCNN) | 53.78 fps ||
| RIFE / rife-v4.6 (TensorRT) | 70.34 fps ||
| CAIN / cvp-v6 (TensorRT) | 45.41 fps ||
| CAIN / cvp-v6 (NCNN) | 10.67 fps ||
| waifu2x / upconv7 (NCNN) | 6.71 fps ||
| RealESRGAN / animevideov3 (TensorRT) | 8.17 fps ||
| DPIR / Denoise (TensorRT) | 4.38 fps ||


# FAQ (Frequently Asked Questions)

> Why do I have to convert engines for CAIN (TensorRT) multiple times?

This is a workaround to make CAIN work on TensorRT inference. It doesn't support dynamic shapes, and you have to convert an engine file for every new video resolution. 
It's a bit inconvenient at first, but definitely worth the 4x-5x speed boost compared to NCNN, especially from the point on once all the engines for your most used resolutions are converted.

> Where can I get custom upscaling models?

I can't include most models in enhancr because of Licensing. Custom models can be obtained [here](https://upscale.wiki/wiki/Model_Database) and loaded inside the models tab.

> Why is Scene Detection in CAIN (TensorRT) so slow when using AV1 input?

It is powered by OpenCV to detect scene changes, which still, as of the time of writing, doesn't support AV1 natively. I tried compiling against a custom ffmpeg build with dav1d, but couldn't get it to compile for Windows after hours of trying around. Recently though there were made some commits to the OpenCV repo for the 5.0 milestone to finally support AV1 but unfortunately only with AOM decoder, which is painfully slow. I compiled that one for now, so that there is AV1 decoding support at all.

> I'm running out of memory when trying to run a TensorRT inference, what should I do?

Either lower concurrent streams in settings or set up tiling in combination with a smaller engine resolution to decrease Video Memory usage.

# Inferences

[TensorRT](https://developer.nvidia.com/tensorrt) is a highly optimized AI inference runtime for NVIDIA GPUs. It uses benchmarking to find the optimal kernel to use for your specific GPU, and there is an extra step to build an engine on the machine you are going to run the AI on. However, the resulting performance is also typically _much much_ better than any PyTorch or NCNN implementation.

[ncnn](https://github.com/Tencent/ncnn) is a high-performance neural network inference computing framework optimized for mobile platforms. Ncnn does not have any third party dependencies. It is cross-platform, and runs faster than all known open source frameworks on most major platforms. It supports NVIDIA, AMD, Intel Graphics and even Apple Silicon.
Ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

# Supporting this project

You can support this project over on [**Patreon**](https://www.patreon.com/mafiosnik) or donate via [**PayPal**](https://www.paypal.com/paypalme/mafiosnik) to speed up development and encourage more releases **and/or** contribute yourself by opening a **pull request**.
I appreciate every form of support towards this project and it means a lot!

# Plans for the future

The code is somewhat messy at some points right now, due to time pressure I stood under, but I'll rewrite affected files one by one with time.
Of course there will be updates with new features and bugfixes regularly and also implementations of new interesting technologies and models when they release.

# Join the discord

To interact with the community, share your results or to get help when encountering any problems visit our [discord](https://discord.gg/jBDqCkSxYz). Previews of upcoming versions are gonna be showcased on there as well. 
