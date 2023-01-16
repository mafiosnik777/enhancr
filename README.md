
# ![heading-icon](https://mafiosnik.needsmental.help/TuuFJcPFi2.png?key=wfGxa75b2G2pyp)

**enhancr** is an **elegant and easy to use** GUI for **Video Frame Interpolation** and **Video Upscaling** which takes advantage of artificial intelligence - built using **node.js** and **Electron**. It was created **to enhance the user experience** for anyone interested in enhancing video footage using artificial intelligence. The GUI was **designed to provide a stunning experience** powered by state-of-the-art technologies **without feeling clunky and outdated** like other alternatives.

![gui-preview-image](https://mafiosnik.needsmental.help/1wLGH7CxEA.png?key=uk7M7TaolACipq)

It features blazing-fast **TensorRT** inference by NVIDIA, which can speed up AI processes **significantly**. **Pre-packaged, without the need to install Docker or WSL** (Windows Subsystem for Linux) - and **NCNN** inference by Tencent which is lightweight and runs on **NVIDIA**, **AMD** and even **Apple Silicon** - in contrast to the mammoth of an inference PyTorch is, which **only runs on NVIDIA GPUs**.

# Features
- Encodes video on the fly and reads frames from source video, without the need of extracting frames or loading into memory
- Queue for batch processing
- Live Preview of output media
- Allows chaining of interpolation, upscaling & restoration
- Offers the possibility to trim videos before processing
- Can load custom ESRGAN models in onnx & pth format and converts them automatically
- Has Scene Detection built-in, to skip interpolation on scene change frames & mitigate artifacts
- Color Themes for user customization
- Discord Rich Presence, to show all your friends progress, current speed & what you're currently enhancing
- ... and much more

## Interpolation

>**RIFE (NCNN)** - [megvii-research](https://github.com/megvii-research)/**[ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)** - powered by [styler00dollar](https://github.com/styler00dollar)/**[VapourSynth-RIFE-NCNN-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-NCNN-Vulkan)**

>**RIFE (TensorRT)** - [megvii-research](https://github.com/megvii-research)/**[ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)** & [styler00dollar](https://github.com/styler00dollar)/**[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)**

>**GMFUpSS (PyTorch)** - [98mxr](https://github.com/98mxr)/**[GMFupSS](https://github.com/98mxr/GMFupSS)** - [styler00dollar](https://github.com/styler00dollar)/**[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)**

>**GMFSS_Union (PyTorch)** - [98mxr](https://github.com/98mxr)/**[GMFSS_Union](https://github.com/98mxr/GMFSS_union)** - powered by [styler00dollar](https://github.com/styler00dollar)/**[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)**

>**CAIN (NCNN)** - [myungsub](https://github.com/myungsub)/**[CAIN](https://github.com/myungsub/CAIN)** - powered by [mafiosnik](https://github.com/mafiosnik777)/**vsynth-cain-NCNN-vulkan** (unreleased)

>**CAIN (TensorRT)** - [myungsub](https://github.com/myungsub)/**[CAIN](https://github.com/myungsub/CAIN)** - powered by [HubertSotnowski](https://github.com/HubertSotnowski)/**[cain-TensorRT](https://github.com/HubertSotnowski/cain-TensorRT)**

*Thanks to [HubertSontowski](https://github.com/HubertSotnowski) and [styler00dollar](https://github.com/styler00dollar) for helping out with implementing CAIN.*

## Upscaling

>**waifu2x (NCNN)** - [nagadomi](https://github.com/nagadomi)/**[waifu2x](https://github.com/nagadomi/waifu2x)** - powered by [nlzy](https://github.com/nlzy)/**[vapoursynth-waifu2x-NCNN-vulkan](https://github.com/nlzy/vapoursynth-waifu2x-NCNN-vulkan)**

>**RealESRGAN (NCNN)** - [xinntao](https://github.com/xinntao)/**[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

>**RealESRGAN (TensorRT)** - [xinntao](https://github.com/xinntao)/**[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** - powered by [AmusementClub](https://github.com/AmusementClub)/**[vs-mlrt](https://github.com/AmusementClub/vs-mlrt)**

*Thanks to [HubertSontowski](https://github.com/HubertSotnowski) for helping out with implementing AnimeSR*

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


To ensure that you have the most recent version of the software and all necessary dependencies, we recommend downloading the installer from [Patreon](https://www.patreon.com/mafiosnik). 
Please note that builds and an embeddable python environment **are not** provided through this repository.

![installer](https://mafiosnik.needsmental.help/mEerVMP8LA.png?key=bzdnzy2RYJGOvO)

>*There probably will be free versions down the line, after a backlog of versions has released.*

# macOS and Linux Support

The GUI was created with cross-platform compatibility in mind and is compatible with both operating systems.
**Our primary focus at the moment is ensuring a stable and fully functioning solution for Windows users, but support for Linux and macOS will be made available in the near future.**

![enhancr-macos](https://mafiosnik.needsmental.help/st6TAP6g9t.png?key=kfq89rqeM2kEdi)

Support for Apple Silicon is planned as well, but I currently only have an Intel Macbook Pro available for testing.

# Benchmarks

Input size: 1920x1080 @ 2x

|| RTX 2060 Super <sup>1</sup> | RTX 3070 <sup>2</sup>| RTX A4000 <sup>3</sup> | RTX 3090 Ti <sup>4</sup> | RTX 4090 <sup>5</sup>
|--|--|--|--|--|--|
| RIFE / rife-v4.6 (NCNN) | 53.78 fps | 64.08 fps | 80.56 fps | 86.24 fps | 136.13 fps |
| RIFE / rife-v4.6 (TensorRT) | 70.34 fps | 94.63 fps | 86.47 fps | 122.68 fps | 170.91 fps |
| CAIN / cvp-v6 (NCNN) | 9.42 fps | 10.56 fps | 13.42 fps | 17.36 fps | 44.87 fps |
| CAIN / cvp-v6 (TensorRT) | 45.41 fps | 63.84 fps | 81.23 fps | 112.87 fps | 183.46 fps |
| GMFSS / Up (PyTorch) | - | - | 4.32 fps | - | - |
| GMFSS / Union (PyTorch) | - | - | 3.68 fps | - | - |
| waifu2x / anime_style_art_rgb (NCNN) | 6.71 fps | 9.36 fps | 9.81 fps | 15.48 fps | 39.77 fps |
| RealESRGAN / animevideov3 (TensorRT) | 7.64 fps | 9.10 fps | 8.49 fps | 18.66 fps | 38.67 fps |
| DPIR / Denoise (TensorRT) | 4.38 fps | 6.45 fps | 5.39 fps | 11.64 fps | 27.41 fps |

<sup>1</sup> <sub>Ryzen 5 3600X - Gainward RTX 2060 Super @ Stock</sub>

<sup>2</sup> <sub>Ryzen 7 3800X - Gigabyte RTX 3070 Eagle OC @ Stock</sub>

<sup>3</sup> <sub>Ryzen 5 3600X - PNY RTX A4000 @ Stock </sub>

<sup>4</sup> <sub>i9 12900KF - ASUS RTX 3090 Ti Strix OC @ ~2220MHz</sub>

<sup>5</sup> <sub>Ryzen 9 5950X - ASUS RTX 4090 Strix OC - @ ~3100MHz with curve to achieve maximum performance</sub>

# Troubleshooting and FAQ (Frequently Asked Questions)

This section has moved to the wiki: https://github.com/mafiosnik777/enhancr/wiki

Check it out to learn more about getting the most out of enhancr or how to fix various problems.

# Inferences

[TensorRT](https://developer.nvidia.com/tensorrt) is a highly optimized AI inference runtime for NVIDIA GPUs. It uses benchmarking to find the optimal kernel to use for your specific GPU, and there is an extra step to build an engine on the machine you are going to run the AI on. However, the resulting performance is also typically _much much_ better than any PyTorch or NCNN implementation.

[NCNN](https://github.com/Tencent/ncnn) is a high-performance neural network inference computing framework optimized for mobile platforms. NCNN does not have any third party dependencies. It is cross-platform, and runs faster than all known open source frameworks on most major platforms. It supports NVIDIA, AMD, Intel Graphics and even Apple Silicon.
NCNN is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

# Supporting this project

I would be grateful if you could show your support for this project by contributing on [Patreon](https://www.patreon.com/mafiosnik) or through a donation on [PayPal](https://www.paypal.com/paypalme/mafiosnik). Your support will help to accelerate development and bring more updates to the project. Additionally, if you have the skills, you can also contribute by opening a pull request. Regardless of the form of support you choose to give, know that it is greatly appreciated.

# Plans for the future

I am continuously working to improve the codebase, including addressing any inconsistencies that may have arisen due to time constraints. Regular updates will be released, including new features, bug fixes, and the incorporation of new technologies and models as they become available. Thank you for your understanding and support.

# Join the discord

To interact with the community, share your results or to get help when encountering any problems visit our [discord](https://discord.gg/jBDqCkSxYz). Previews of upcoming versions are gonna be showcased on there as well. 
