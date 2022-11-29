const electron = require("electron");
const ipc = electron.ipcRenderer;
var path = require("path");
const fs = require("fs");

var currentProject = sessionStorage.getItem("currentProject");
var currentProjectSpan = document.getElementById("current-project");
var content = document.getElementById("content");

var videoInputText = document.getElementById("input-video-text");

var outputPathText = document.getElementById("output-path-text");
var ffmpegParams = document.getElementById("ffmpeg-params");

var x264Btn = document.getElementById("x264");
var x265Btn = document.getElementById("x265");
var AV1Btn = document.getElementById("AV1");
var VP9Btn = document.getElementById("VP9");
var ProResBtn = document.getElementById("ProRes");
var NVENCBtn = document.getElementById("NVENC");

var outputContainerSpan = document.getElementById("container-span");
var modelSpan = document.getElementById("model-span");

// load project name
currentProjectSpan.textContent = path.basename(currentProject, ".enhncr");

function loadInterpolation() {
  // read data from project file
  const data = JSON.parse(fs.readFileSync(currentProject));

  // fill video input
  if (
    data.interpolation[0].inputFile.length >= 55 &&
    path.basename(data.interpolation[0].inputFile).length >= 55
  ) {
    videoInputText.textContent =
      "../" +
      path.basename(data.interpolation[0].inputFile).substr(0, 55) +
      "\u2026";
  } else if (data.interpolation[0].inputFile.length >= 55) {
    videoInputText.textContent =
      "../" + path.basename(data.interpolation[0].inputFile);
  } else {
    videoInputText.textContent = data.interpolation[0].inputFile;
  }
  sessionStorage.setItem("inputPath", data.interpolation[0].inputFile);
  // fill output path
  outputPathText.textContent = data.interpolation[0].outputPath;
  sessionStorage.setItem("outputPath", data.interpolation[0].outputPath);
  // fill codec && ffmpeg params
  if (data.interpolation[0].codec === "H264") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecInterpolation', 'x264');

    ffmpegParams.value = jsonCodec.codecs[0].x264;

    x265Btn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    NVENCBtn.style.color = "#d0d0d0";
    x264Btn.style.color = "#1e5cce";
  }
  if (data.interpolation[0].codec === "H265") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecInterpolation', 'x265');

    ffmpegParams.value = jsonCodec.codecs[0].x265;

    x264Btn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    NVENCBtn.style.color = "#d0d0d0";
    x265Btn.style.color = "#1e5cce";
  }
  if (data.interpolation[0].codec === "VP9") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var json = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecInterpolation', 'VP9');

    ffmpegParams.value = json.codecs[0].VP9;

    x264Btn.style.color = "#d0d0d0";
    x265Btn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    NVENCBtn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#1e5cce";
  }
  if (data.interpolation[0].codec === "AV1") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecInterpolation', 'AV1');

    ffmpegParams.value = jsonCodec.codecs[0].AV1;

    x264Btn.style.color = "#d0d0d0";
    x265Btn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    NVENCBtn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#1e5cce";
  }
  if (data.interpolation[0].codec === "ProRes") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    ffmpegParams.value = jsonCodec.codecs[0].ProRes;

    sessionStorage.setItem('codecInterpolation', 'ProRes');

    x264Btn.style.color = "#d0d0d0";
    x265Btn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#1e5cce";
  }
  if (data.interpolation[0].codec === "NVENC") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecInterpolation', 'NVENC');

    ffmpegParams.value = jsonCodec.codecs[0].NVENC;

    x264Btn.style.color = "#d0d0d0";
    x265Btn.style.color = "#d0d0d0";
    AV1Btn.style.color = "#d0d0d0";
    ProResBtn.style.color = "#d0d0d0";
    VP9Btn.style.color = "#d0d0d0";
    NVENCBtn.style.color = "#1e5cce";
  }
  if (!data.interpolation[0].params == "") {
    ffmpegParams.value = data.interpolation[0].params;
  }
  // fill output container input
  if (data.interpolation[0].outputContainer === "") {
    outputContainerSpan.textContent = ".mkv";
  } else {
    outputContainerSpan.textContent = data.interpolation[0].outputContainer;
  }

  var rife23Option = document.getElementById("rife-23");
  var rife4Option = document.getElementById("rife-4");
  var rife46Option = document.getElementById("rife-46");
  var rvpv1Option = document.getElementById("rvp-v1");
  var cvpv6Option = document.getElementById("cvp-v6");

  var interpolationEngineSpan = document.getElementById("interpolation-engine-text");
  // fill engine input
  if (data.interpolation[0].engine === "cain") {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (NCNN)";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
  }
  if (data.interpolation[0].engine === "cain-trt") {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (TensorRT)";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
  }
  if (data.interpolation[0].engine === "rife") {
    interpolationEngineSpan.textContent = "Optical Flow - RIFE (NCNN)";
    rvpv1Option.style.display = 'none';
    cvpv6Option.style.display = 'none';
  }
  if (data.interpolation[0].engine === "rife-trt") {
    interpolationEngineSpan.textContent = "Optical Flow - RIFE (TensorRT)";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rvpv1Option.style.display = 'none';
    cvpv6Option.style.display = 'none';
  }
  if (data.interpolation[0].engine === "") {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (TensorRT)";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
  }

  // fill model input
  if (data.interpolation[0].model === "") {
    modelSpan.textContent = "CVP - v6.0";
  } else {
    modelSpan.textContent = data.interpolation[0].model;
  }
}
loadInterpolation();

var videoInputTextUpscale = document.getElementById("upscale-input-text"),
  outputPathTextUpscale = document.getElementById("upscale-output-path-text"),
  ffmpegParamsUpscale = document.getElementById("ffmpeg-params-upscale");

var x265BtnUp = document.getElementById("x265-up"),
  AV1BtnUp = document.getElementById("AV1-up"),
  VP9BtnUp = document.getElementById("VP9-up"),
  ProResBtnUp = document.getElementById("ProRes-up"),
  NVENCBtnUp = document.getElementById("NVENC-up"),
  x264BtnUp = document.getElementById("x264-up");

var outputContainerSpanUpscale = document.getElementById("container-span-up"),
  scaleSpan = document.getElementById("factor-span");

function loadUpscaling() {
  // read data from project file
  const data = JSON.parse(fs.readFileSync(currentProject));

  // fill video input
  if (
    data.upscaling[0].inputFile.length >= 55 &&
    path.basename(data.upscaling[0].inputFile).length >= 55
  ) {
    videoInputTextUpscale.textContent =
      "../" +
      path.basename(data.upscaling[0].inputFile).substr(0, 55) +
      "\u2026";
  } else if (data.upscaling[0].inputFile.length >= 55) {
    videoInputTextUpscale.textContent =
      "../" + path.basename(data.upscaling[0].inputFile);
  } else {
    videoInputTextUpscale.textContent = data.upscaling[0].inputFile;
  }
  sessionStorage.setItem("upscaleInputPath", data.upscaling[0].inputFile);

  // fill output path
  outputPathTextUpscale.textContent = data.upscaling[0].outputPath;
  sessionStorage.setItem("upscaleOutputPath", data.upscaling[0].outputPath);

  // fill codec && ffmpeg params
  if (data.upscaling[0].codec === "H264") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'x264');

    ffmpegParamsUpscale.value = jsonCodec.codecs[0].x264;

    x265BtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    NVENCBtnUp.style.color = "#d0d0d0";
    x264BtnUp.style.color = "#1e5cce";
  }
  if (data.upscaling[0].codec === "H265") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'x265');

    ffmpegParamsUpscale.value = jsonCodec.codecs[0].x265;

    x264BtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    NVENCBtnUp.style.color = "#d0d0d0";
    x265BtnUp.style.color = "#1e5cce";
  }
  if (data.upscaling[0].codec === "VP9") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var json = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'VP9');

    ffmpegParamsUpscale.value = json.codecs[0].VP9;

    x264BtnUp.style.color = "#d0d0d0";
    x265BtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    NVENCBtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#1e5cce";
  }
  if (data.upscaling[0].codec === "AV1") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'AV1');

    ffmpegParamsUpscale.value = jsonCodec.codecs[0].AV1;

    x264BtnUp.style.color = "#d0d0d0";
    x265BtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    NVENCBtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#1e5cce";
  }

  if (data.upscaling[0].codec === "ProRes") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'ProRes');

    ffmpegParamsUpscale.value = jsonCodec.codecs[0].ProRes;

    x264BtnUp.style.color = "#d0d0d0";
    x265BtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#1e5cce";
  }

  if (data.upscaling[0].codec === "NVENC") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecUpscaling', 'NVENC');

    ffmpegParamsUpscale.value = jsonCodec.codecs[0].NVENC;

    x264BtnUp.style.color = "#d0d0d0";
    x265BtnUp.style.color = "#d0d0d0";
    AV1BtnUp.style.color = "#d0d0d0";
    ProResBtnUp.style.color = "#d0d0d0";
    VP9BtnUp.style.color = "#d0d0d0";
    NVENCBtnUp.style.color = "#1e5cce";
  }

  if (!data.upscaling[0].params == "") {
    ffmpegParams.value = data.upscaling[0].params;
  }

  // fill output container input
  if (data.upscaling[0].outputContainer === "") {
    outputContainerSpanUpscale.textContent = ".mkv";
  } else {
    outputContainerSpanUpscale.textContent = data.upscaling[0].outputContainer;
  }

  var upscaleEngineSpan = document.getElementById("upscale-engine-text");
  // fill engine input
  if (data.upscaling[0].engine === "waifu2x") {
    upscaleEngineSpan.textContent = "Upscaling - waifu2x (NCNN)"
  }
  if (data.upscaling[0].engine === "realesrgan") {
    upscaleEngineSpan.textContent = "Upscaling - RealESRGAN (TensorRT)"
  }
  if (data.upscaling[0].engine === "animesr") {
    upscaleEngineSpan.textContent = "Upscaling - AnimeSR (TensorRT)"
  }
}
loadUpscaling();

const videoInputTextRes = document.getElementById('restore-input-text');
const outputPathTextRes = document.getElementById('restore-output-path-text');

const ffmpegParamsRes = document.getElementById('ffmpeg-params-restoration');

var x265BtnRes = document.getElementById("x265-res"),
  AV1BtnRes = document.getElementById("AV1-res"),
  VP9BtnRes = document.getElementById("VP9-res"),
  ProResBtnRes = document.getElementById("ProRes-res"),
  NVENCBtnRes = document.getElementById("NVENC-res"),
  x264BtnRes = document.getElementById("x264-res");

function loadRestoration() {
  // read data from project file
  const data = JSON.parse(fs.readFileSync(currentProject));

  // fill video input
  if (
    data.restoration[0].inputFile.length >= 55 &&
    path.basename(data.restoration[0].inputFile).length >= 55
  ) {
    videoInputTextRes.textContent =
      "../" +
      path.basename(data.restoration[0].inputFile).substr(0, 55) +
      "\u2026";
  } else if (data.restoration[0].inputFile.length >= 55) {
    videoInputTextRes.textContent =
      "../" + path.basename(data.restoration[0].inputFile);
  } else {
    videoInputTextRes.textContent = data.restoration[0].inputFile;
  }
  sessionStorage.setItem("inputPathRestore", data.restoration[0].inputFile);
  // fill output path
  outputPathTextRes.textContent = data.restoration[0].outputPath;
  sessionStorage.setItem("outputPathRestoration", data.restoration[0].outputPath);
  // fill codec && ffmpeg params
  if (data.restoration[0].codec === "H264") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecRestoration', 'x264');

    ffmpegParamsRes.value = jsonCodec.codecs[0].x264;

    x265BtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    NVENCBtnRes.style.color = "#d0d0d0";
    x264BtnRes.style.color = "#1e5cce";
  }
  if (data.restoration[0].codec === "H265") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecRestoration', 'x265');

    ffmpegParamsRes.value = jsonCodec.codecs[0].x265;

    x264BtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    NVENCBtnRes.style.color = "#d0d0d0";
    x265BtnRes.style.color = "#1e5cce";
  }
  if (data.restoration[0].codec === "VP9") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var json = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecRestoration', 'VP9');

    ffmpegParamsRes.value = json.codecs[0].VP9;

    x264BtnRes.style.color = "#d0d0d0";
    x265BtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    NVENCBtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#1e5cce";
  }
  if (data.restoration[0].codec === "AV1") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecRestoration', 'AV1');

    ffmpegParamsRes.value = jsonCodec.codecs[0].AV1;

    x264BtnRes.style.color = "#d0d0d0";
    x265BtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    NVENCBtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#1e5cce";
  }
  if (data.restoration[0].codec === "ProRes") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    ffmpegParamsRes.value = jsonCodec.codecs[0].ProRes;

    sessionStorage.setItem('codecRestoration', 'ProRes');

    x264BtnRes.style.color = "#d0d0d0";
    x265BtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#1e5cce";
  }
  if (data.restoration[0].codec === "NVENC") {
    try {
      const jsonString = fs.readFileSync(path.join(__dirname, "codecs.json"));
      var jsonCodec = JSON.parse(jsonString);
    } catch (err) {
      console.log(err);
      return;
    }

    sessionStorage.setItem('codecRestoration', 'NVENC');

    ffmpegParamsRes.value = jsonCodec.codecs[0].NVENC;

    x264BtnRes.style.color = "#d0d0d0";
    x265BtnRes.style.color = "#d0d0d0";
    AV1BtnRes.style.color = "#d0d0d0";
    ProResBtnRes.style.color = "#d0d0d0";
    VP9BtnRes.style.color = "#d0d0d0";
    NVENCBtnRes.style.color = "#1e5cce";
  }

  if (!data.restoration[0].params == "") {
    ffmpegParams.value = data.restoration[0].params;
  }

  const outputContainerSpanRes = document.getElementById('container-span-res');
  // fill output container input
  if (data.restoration[0].outputContainer === "") {
    outputContainerSpanRes.textContent = ".mkv";
  } else {
    outputContainerSpanRes.textContent = data.restoration[0].outputContainer;
  }

  var restorationEngineSpan = document.getElementById("restoration-engine-text");
  // fill engine input
  if (data.restoration[0].engine === "dpir") {
    restorationEngineSpan.textContent = "Restoration - DPIR (TensorRT)";
  } else {
    restorationEngineSpan.textContent = "Restoration - AnimeVideo (TensorRT)";
  }

  const modelSpanRes = document.getElementById('model-span-restoration');

  const denoiseSharpen = document.getElementById('denoise-sharpen');
  const denoise = document.getElementById('denoise');
  const deblock = document.getElementById('deblock');

  // fill model input
  if (data.restoration[0].engine === "dpir") {
    modelSpanRes.textContent = "Denoise";
    denoiseSharpen.style.display = 'none';
    denoise.style.display = 'block';
    deblock.style.display = 'block';
  } else {
    modelSpanRes.textContent = "Denoise/Sharpen";
    denoiseSharpen.style.display = 'block';
    denoise.style.display = 'none';
    deblock.style.display = 'none';
  }
}
loadRestoration();

var mediaInfoContainer = document.getElementById("mediainfo-container");
var outputContainerSpan = document.getElementById("container-span");
mediaInfoContainer.textContent = outputContainerSpan.textContent;

var mediaInfoContainerUp = document.getElementById("mediainfo-containerUp");
var outputContainerSpanUp = document.getElementById("container-span-up");
mediaInfoContainerUp.textContent = outputContainerSpanUp.textContent;

var mediaInfoContainerRes = document.getElementById("mediainfo-containerRes");
var outputContainerSpanRes = document.getElementById("container-span-res");
mediaInfoContainerRes.textContent = outputContainerSpanRes.textContent;

