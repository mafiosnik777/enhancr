const fs = require("fs");
const path = require("path");

// color theme selector
var selector = document.getElementById("theme-selector");

// ui elements
var settingsSaveButton = document.getElementById("settings-button-text");
var interpolateButton = document.getElementById("interpolate-button-text");
var upscalingButton = document.getElementById("upscaling-button-text");
var restoreButton = document.getElementById('restore-button-text');
var settingsSwitcher = document.getElementById("settings-switcher");

var toggleRpc = document.getElementById("toggle-rpc");
var toggleBlur = document.getElementById("disable-blur");
var togglePreview = document.getElementById("preview-check");
var toggleOled = document.getElementById("oled-mode");
var toggleRifeTta = document.getElementById("rife-tta-check");
var toggleRifeUhd = document.getElementById("rife-uhd-check");
var scCheck = document.getElementById("sc-check");
var skipCheck = document.getElementById("skip-check");
var togglefp16 = document.getElementById("fp16-check");
var toggleShapes = document.getElementById('shape-check');
var toggleTiling = document.getElementById('tiling-check');
var customModelCheck = document.getElementById('custom-model-check');
var pythonCheck = document.getElementById('python-check');

var mediaInfoText = document.getElementsByClassName("info-text");
var tooltips = document.getElementsByClassName("tooltip-text");
var header = document.getElementsByClassName("theme-header");
var side = document.getElementsByClassName("theme-side");

var blue = document.getElementById("theme-select-blue");

var progressDone = document.getElementById("progress-done");


let queueProgressBar = document.getElementsByClassName('queue-item-progress-overlay');

function blueSelect() {
    selector.style.top = "38%";
    selector.style.left = "13%";
    sessionStorage.setItem('theme', 'blue');
    settingsSaveButton.style.color = "#1e5cce";
    interpolateButton.style.color = "#1e5cce";
    upscalingButton.style.color = "#1e5cce";
    restoreButton.style.color = "#1e5cce";
    settingsSwitcher.style.color = "#1e5cce";

    toggleRpc.style.setProperty('--toggle-color', "#1e5cce");
    toggleBlur.style.setProperty('--toggle-color', "#1e5cce");
    togglePreview.style.setProperty('--toggle-color', "#1e5cce");
    toggleOled.style.setProperty('--toggle-color', "#1e5cce");
    toggleRifeTta.style.setProperty('--toggle-color', "#1e5cce");
    toggleRifeUhd.style.setProperty('--toggle-color', "#1e5cce");
    togglefp16.style.setProperty('--toggle-color', "#1e5cce");
    toggleShapes.style.setProperty('--toggle-color', "#1e5cce");
    toggleTiling.style.setProperty('--toggle-color', "#1e5cce");
    customModelCheck.style.setProperty('--toggle-color', "#1e5cce");
    scCheck.style.setProperty('--toggle-color', "#1e5cce");
    skipCheck.style.setProperty('--toggle-color', "#1e5cce");
    pythonCheck.style.setProperty('--toggle-color', "#1e5cce");
    progressDone.style.setProperty('--bar-color', "#1e5cce");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#1e5cce";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#1e5cce");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#1c73d333";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#1e5cce";
    codecInterpolation.style.color = "#1e5cce";
    codecUpscaling.style.color = "#1e5cce";

    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#1e5cce";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#1e5cce";
        tooltips[i].style.setProperty('--tooltip-triangle', "#1e5cce");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
blue.addEventListener("click", blueSelect);

var pink = document.getElementById("theme-select-pink");

function pinkSelect() {
    selector.style.top = "38%";
    selector.style.left = "25%";
    sessionStorage.setItem('theme', 'pink');
    settingsSaveButton.style.color = "#ce1e6c";
    interpolateButton.style.color = "#ce1e6c";
    upscalingButton.style.color = "#ce1e6c";
    restoreButton.style.color = "#ce1e6c";
    settingsSwitcher.style.color = "#ce1e6c";

    toggleRpc.style.setProperty('--toggle-color', "#ce1e6c");
    toggleBlur.style.setProperty('--toggle-color', "#ce1e6c");
    togglePreview.style.setProperty('--toggle-color', "#ce1e6c");
    toggleOled.style.setProperty('--toggle-color', "#ce1e6c");
    toggleRifeTta.style.setProperty('--toggle-color', "#ce1e6c");
    toggleRifeUhd.style.setProperty('--toggle-color', "#ce1e6c");
    togglefp16.style.setProperty('--toggle-color', "#ce1e6c");
    toggleShapes.style.setProperty('--toggle-color', "#ce1e6c");
    toggleTiling.style.setProperty('--toggle-color', "#ce1e6c");
    customModelCheck.style.setProperty('--toggle-color', "#ce1e6c");
    pythonCheck.style.setProperty('--toggle-color', "#ce1e6c");
    progressDone.style.setProperty('--bar-color', "#ce1e6c");
    scCheck.style.setProperty('--toggle-color', "#ce1e6c");
    skipCheck.style.setProperty('--toggle-color', "#ce1e6c");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#ce1e6c";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#ce1e6c");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#ce1e6c33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#ce1e6c";
    codecInterpolation.style.color = "#ce1e6c";
    codecUpscaling.style.color = "#ce1e6c";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#ce1e6c";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#ce1e6c";
        tooltips[i].style.setProperty('--tooltip-triangle', "#ce1e6c");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
pink.addEventListener("click", pinkSelect);

var green = document.getElementById("theme-select-green");

function greenSelect() {
    selector.style.top = "38%";
    selector.style.left = "37%";
    sessionStorage.setItem('theme', 'green');
    settingsSaveButton.style.color = "#9ece1e";
    interpolateButton.style.color = "#9ece1e";
    upscalingButton.style.color = "#9ece1e";
    restoreButton.style.color = "#9ece1e";
    settingsSwitcher.style.color = "#9ece1e";

    toggleRpc.style.setProperty('--toggle-color', "#9ece1e");
    toggleBlur.style.setProperty('--toggle-color', "#9ece1e");
    togglePreview.style.setProperty('--toggle-color', "#9ece1e");
    toggleOled.style.setProperty('--toggle-color', "#9ece1e");
    toggleRifeTta.style.setProperty('--toggle-color', "#9ece1e");
    toggleRifeUhd.style.setProperty('--toggle-color', "#9ece1e");
    togglefp16.style.setProperty('--toggle-color', "#9ece1e");
    toggleShapes.style.setProperty('--toggle-color', "#9ece1e");
    toggleTiling.style.setProperty('--toggle-color', "#9ece1e");
    customModelCheck.style.setProperty('--toggle-color', "#9ece1e");
    pythonCheck.style.setProperty('--toggle-color', "#9ece1e");
    progressDone.style.setProperty('--bar-color', "#9ece1e");
    scCheck.style.setProperty('--toggle-color', "#9ece1e");
    skipCheck.style.setProperty('--toggle-color', "#9ece1e");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#9ece1e";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#9ece1e");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#9ece1e33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#9ece1e";
    codecInterpolation.style.color = "#9ece1e";
    codecUpscaling.style.color = "#9ece1e";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#9ece1e";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#9ece1e";
        tooltips[i].style.setProperty('--tooltip-triangle', "#9ece1e");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
green.addEventListener("click", greenSelect);

var purple = document.getElementById("theme-select-purple");

function purpleSelect() {
    selector.style.top = "38%";
    selector.style.left = "49%";
    sessionStorage.setItem('theme', 'purple');
    settingsSaveButton.style.color = "#601ece";
    interpolateButton.style.color = "#601ece";
    upscalingButton.style.color = "#601ece";
    restoreButton.style.color = "#601ece";
    settingsSwitcher.style.color = "#601ece";

    toggleRpc.style.setProperty('--toggle-color', "#601ece");
    toggleBlur.style.setProperty('--toggle-color', "#601ece");
    togglePreview.style.setProperty('--toggle-color', "#601ece");
    toggleOled.style.setProperty('--toggle-color', "#601ece");
    toggleRifeTta.style.setProperty('--toggle-color', "#601ece");
    toggleRifeUhd.style.setProperty('--toggle-color', "#601ece");
    togglefp16.style.setProperty('--toggle-color', "#601ece");
    toggleShapes.style.setProperty('--toggle-color', "#601ece");
    toggleTiling.style.setProperty('--toggle-color', "#601ece");
    customModelCheck.style.setProperty('--toggle-color', "#601ece");
    pythonCheck.style.setProperty('--toggle-color', "#601ece");
    progressDone.style.setProperty('--bar-color', "#601ece");
    scCheck.style.setProperty('--toggle-color', "#601ece");
    skipCheck.style.setProperty('--toggle-color', "#601ece");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#601ece";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#601ece");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#601ece33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#601ece";
    codecInterpolation.style.color = "#601ece";
    codecUpscaling.style.color = "#601ece";

    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#601ece";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#601ece";
        tooltips[i].style.setProperty('--tooltip-triangle', "#601ece");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
purple.addEventListener("click", purpleSelect);

var orange = document.getElementById("theme-select-orange");

function orangeSelect() {
    selector.style.top = "38%";
    selector.style.left = "61%";
    sessionStorage.setItem('theme', 'orange');
    settingsSaveButton.style.color = "#e36812";
    interpolateButton.style.color = "#e36812";
    upscalingButton.style.color = "#e36812";
    restoreButton.style.color = "#e36812";
    settingsSwitcher.style.color = "#e36812";

    toggleRpc.style.setProperty('--toggle-color', "#e36812");
    toggleBlur.style.setProperty('--toggle-color', "#e36812");
    togglePreview.style.setProperty('--toggle-color', "#e36812");
    toggleOled.style.setProperty('--toggle-color', "#e36812");
    toggleRifeTta.style.setProperty('--toggle-color', "#e36812");
    toggleRifeUhd.style.setProperty('--toggle-color', "#e36812");
    togglefp16.style.setProperty('--toggle-color', "#e36812");
    toggleShapes.style.setProperty('--toggle-color', "#e36812");
    toggleTiling.style.setProperty('--toggle-color', "#e36812");
    customModelCheck.style.setProperty('--toggle-color', "#e36812");
    pythonCheck.style.setProperty('--toggle-color', "#e36812");
    progressDone.style.setProperty('--bar-color', "#e36812");
    scCheck.style.setProperty('--toggle-color', "#e36812");
    skipCheck.style.setProperty('--toggle-color', "#e36812");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#e36812";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#e36812");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#e3681233";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#e36812";
    codecInterpolation.style.color = "#e36812";
    codecUpscaling.style.color = "#e36812";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#e36812";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#e36812";
        tooltips[i].style.setProperty('--tooltip-triangle', "#e36812");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
orange.addEventListener("click", orangeSelect);

var yellow = document.getElementById("theme-select-yellow");

function yellowSelect() {
    selector.style.top = "38%";
    selector.style.left = "73%";
    sessionStorage.setItem('theme', 'yellow');
    settingsSaveButton.style.color = "#cece1b";
    interpolateButton.style.color = "#cece1b";
    upscalingButton.style.color = "#cece1b";
    restoreButton.style.color = "#cece1b";
    settingsSwitcher.style.color = "#cece1b";

    toggleRpc.style.setProperty('--toggle-color', "#cece1b");
    toggleBlur.style.setProperty('--toggle-color', "#cece1b");
    togglePreview.style.setProperty('--toggle-color', "#cece1b");
    toggleOled.style.setProperty('--toggle-color', "#cece1b");
    toggleRifeTta.style.setProperty('--toggle-color', "#cece1b");
    toggleRifeUhd.style.setProperty('--toggle-color', "#cece1b");
    togglefp16.style.setProperty('--toggle-color', "#cece1b");
    toggleShapes.style.setProperty('--toggle-color', "#cece1b");
    toggleTiling.style.setProperty('--toggle-color', "#cece1b");
    customModelCheck.style.setProperty('--toggle-color', "#cece1b");
    pythonCheck.style.setProperty('--toggle-color', "#cece1b");
    progressDone.style.setProperty('--bar-color', "#cece1b");
    scCheck.style.setProperty('--toggle-color', "#cece1b");
    skipCheck.style.setProperty('--toggle-color', "#cece1b");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#cece1b";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#cece1b");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#cece1b33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#cece1b";
    codecInterpolation.style.color = "#cece1b";
    codecUpscaling.style.color = "#cece1b";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#cece1b";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#cece1b";
        tooltips[i].style.setProperty('--tooltip-triangle', "#cece1b");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
yellow.addEventListener("click", yellowSelect);

var red = document.getElementById("theme-select-red");

function redSelect() {
    selector.style.top = "38%";
    selector.style.left = "85%";
    sessionStorage.setItem('theme', 'red');
    settingsSaveButton.style.color = "#ce2a1e";
    interpolateButton.style.color = "#ce2a1e";
    upscalingButton.style.color = "#ce2a1e";
    restoreButton.style.color = "#ce2a1e";
    settingsSwitcher.style.color = "#ce2a1e";

    toggleRpc.style.setProperty('--toggle-color', "#ce2a1e");
    toggleBlur.style.setProperty('--toggle-color', "#ce2a1e");
    togglePreview.style.setProperty('--toggle-color', "#ce2a1e");
    toggleOled.style.setProperty('--toggle-color', "#ce2a1e");
    toggleRifeTta.style.setProperty('--toggle-color', "#ce2a1e");
    toggleRifeUhd.style.setProperty('--toggle-color', "#ce2a1e");
    togglefp16.style.setProperty('--toggle-color', "#ce2a1e");
    toggleShapes.style.setProperty('--toggle-color', "#ce2a1e");
    toggleTiling.style.setProperty('--toggle-color', "#ce2a1e");
    customModelCheck.style.setProperty('--toggle-color', "#ce2a1e");
    pythonCheck.style.setProperty('--toggle-color', "#ce2a1e");
    progressDone.style.setProperty('--bar-color', "#ce2a1e");
    scCheck.style.setProperty('--toggle-color', "#ce2a1e");
    skipCheck.style.setProperty('--toggle-color', "#ce2a1e");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#ce2a1e";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#ce2a1e");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#ce2a1e33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#ce2a1e";
    codecInterpolation.style.color = "#ce2a1e";
    codecUpscaling.style.color = "#ce2a1e";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#ce2a1e";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#ce2a1e";
        tooltips[i].style.setProperty('--tooltip-triangle', "#ce2a1e");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
red.addEventListener("click", redSelect);

var sand = document.getElementById("theme-select-sand");

function sandSelect() {
    selector.style.top = "47%";
    selector.style.left = "13%";
    sessionStorage.setItem('theme', 'sand');
    settingsSaveButton.style.color = "#E9DAC1";
    interpolateButton.style.color = "#E9DAC1";
    upscalingButton.style.color = "#E9DAC1";
    restoreButton.style.color = "#E9DAC1";
    settingsSwitcher.style.color = "#E9DAC1";

    toggleRpc.style.setProperty('--toggle-color', "#E9DAC1");
    toggleBlur.style.setProperty('--toggle-color', "#E9DAC1");
    togglePreview.style.setProperty('--toggle-color', "#E9DAC1");
    toggleOled.style.setProperty('--toggle-color', "#E9DAC1");
    toggleRifeTta.style.setProperty('--toggle-color', "#E9DAC1");
    toggleRifeUhd.style.setProperty('--toggle-color', "#E9DAC1");
    togglefp16.style.setProperty('--toggle-color', "#E9DAC1");
    toggleShapes.style.setProperty('--toggle-color', "#E9DAC1");
    toggleTiling.style.setProperty('--toggle-color', "#E9DAC1");
    customModelCheck.style.setProperty('--toggle-color', "#E9DAC1");
    pythonCheck.style.setProperty('--toggle-color', "#E9DAC1");
    progressDone.style.setProperty('--bar-color', "#E9DAC1");
    scCheck.style.setProperty('--toggle-color', "#E9DAC1");
    skipCheck.style.setProperty('--toggle-color', "#E9DAC1");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#E9DAC1";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#E9DAC1");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#E9DAC133";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#E9DAC1";
    codecInterpolation.style.color = "#E9DAC1";
    codecUpscaling.style.color = "#E9DAC1";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#E9DAC1";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#E9DAC1";
        tooltips[i].style.setProperty('--tooltip-triangle', "#E9DAC1");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
sand.addEventListener("click", sandSelect);

var mint = document.getElementById("theme-select-mint");

function mintSelect() {
    selector.style.top = "47%";
    selector.style.left = "25%";
    sessionStorage.setItem('theme', 'mint');
    settingsSaveButton.style.color = "#8FE3CF";
    interpolateButton.style.color = "#8FE3CF";
    upscalingButton.style.color = "#8FE3CF";
    restoreButton.style.color = "#8FE3CF";
    settingsSwitcher.style.color = "#8FE3CF";

    toggleRpc.style.setProperty('--toggle-color', "#8FE3CF");
    toggleBlur.style.setProperty('--toggle-color', "#8FE3CF");
    togglePreview.style.setProperty('--toggle-color', "#8FE3CF");
    toggleOled.style.setProperty('--toggle-color', "#8FE3CF");
    toggleRifeTta.style.setProperty('--toggle-color', "#8FE3CF");
    toggleRifeUhd.style.setProperty('--toggle-color', "#8FE3CF");
    togglefp16.style.setProperty('--toggle-color', "#8FE3CF");
    toggleShapes.style.setProperty('--toggle-color', "#8FE3CF");
    toggleTiling.style.setProperty('--toggle-color', "#8FE3CF");
    customModelCheck.style.setProperty('--toggle-color', "#8FE3CF");
    pythonCheck.style.setProperty('--toggle-color', "#8FE3CF");
    progressDone.style.setProperty('--bar-color', "#8FE3CF");
    scCheck.style.setProperty('--toggle-color', "#8FE3CF");
    skipCheck.style.setProperty('--toggle-color', "#8FE3CF");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#8FE3CF";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#8FE3CF");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#8FE3CF33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#8FE3CF";
    codecInterpolation.style.color = "#8FE3CF";
    codecUpscaling.style.color = "#8FE3CF";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#8FE3CF";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#8FE3CF";
        tooltips[i].style.setProperty('--tooltip-triangle', "#8FE3CF");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
mint.addEventListener("click", mintSelect);

var salmon = document.getElementById("theme-select-salmon");

function salmonSelect() {
    selector.style.top = "47%";
    selector.style.left = "37%";
    sessionStorage.setItem('theme', 'salmon');
    settingsSaveButton.style.color = "#FFB3B3";
    interpolateButton.style.color = "#FFB3B3";
    upscalingButton.style.color = "#FFB3B3";
    restoreButton.style.color = "#FFB3B3";
    settingsSwitcher.style.color = "#FFB3B3";

    toggleRpc.style.setProperty('--toggle-color', "#FFB3B3");
    toggleBlur.style.setProperty('--toggle-color', "#FFB3B3");
    togglePreview.style.setProperty('--toggle-color', "#FFB3B3");
    toggleOled.style.setProperty('--toggle-color', "#FFB3B3");
    toggleRifeTta.style.setProperty('--toggle-color', "#FFB3B3");
    toggleRifeUhd.style.setProperty('--toggle-color', "#FFB3B3");
    togglefp16.style.setProperty('--toggle-color', "#FFB3B3");
    toggleShapes.style.setProperty('--toggle-color', "#FFB3B3");
    toggleTiling.style.setProperty('--toggle-color', "#FFB3B3");
    customModelCheck.style.setProperty('--toggle-color', "#FFB3B3");
    pythonCheck.style.setProperty('--toggle-color', "#FFB3B3");
    progressDone.style.setProperty('--bar-color', "#FFB3B3");
    scCheck.style.setProperty('--toggle-color', "#FFB3B3");
    skipCheck.style.setProperty('--toggle-color', "#FFB3B3");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#FFB3B3";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#FFB3B3");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#FFB3B333";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#FFB3B3";
    codecInterpolation.style.color = "#FFB3B3";
    codecUpscaling.style.color = "#FFB3B3";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#FFB3B3";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#FFB3B3";
        tooltips[i].style.setProperty('--tooltip-triangle', "#FFB3B3");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
salmon.addEventListener("click", salmonSelect);

var egg = document.getElementById("theme-select-egg");

function eggSelect() {
    selector.style.top = "47%";
    selector.style.left = "49%";
    sessionStorage.setItem('theme', 'egg');
    settingsSaveButton.style.color = "#FFEF82";
    interpolateButton.style.color = "#FFEF82";
    upscalingButton.style.color = "#FFEF82";
    restoreButton.style.color = "#FFEF82";
    settingsSwitcher.style.color = "#FFEF82";

    toggleRpc.style.setProperty('--toggle-color', "#FFEF82");
    toggleBlur.style.setProperty('--toggle-color', "#FFEF82");
    togglePreview.style.setProperty('--toggle-color', "#FFEF82");
    toggleOled.style.setProperty('--toggle-color', "#FFEF82");
    toggleRifeTta.style.setProperty('--toggle-color', "#FFEF82");
    toggleRifeUhd.style.setProperty('--toggle-color', "#FFEF82");
    togglefp16.style.setProperty('--toggle-color', "#FFEF82");
    toggleShapes.style.setProperty('--toggle-color', "#FFEF82");
    toggleTiling.style.setProperty('--toggle-color', "#FFEF82");
    customModelCheck.style.setProperty('--toggle-color', "#FFEF82");
    pythonCheck.style.setProperty('--toggle-color', "#FFEF82");
    progressDone.style.setProperty('--bar-color', "#FFEF82");
    scCheck.style.setProperty('--toggle-color', "#FFEF82");
    skipCheck.style.setProperty('--toggle-color', "#FFEF82");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#FFEF82";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#FFEF82");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#FFEF8233";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#FFEF82";
    codecInterpolation.style.color = "#FFEF82";
    codecUpscaling.style.color = "#FFEF82";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#FFEF82";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#FFEF82";
        tooltips[i].style.setProperty('--tooltip-triangle', "#FFEF82");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
egg.addEventListener("click", eggSelect);

var lavender = document.getElementById("theme-select-lavender");

function lavenderSelect() {
    selector.style.top = "47%";
    selector.style.left = "61%";
    sessionStorage.setItem('theme', 'lavender');
    settingsSaveButton.style.color = "#AFB4FF";
    interpolateButton.style.color = "#AFB4FF";
    upscalingButton.style.color = "#AFB4FF";
    restoreButton.style.color = "#AFB4FF";
    settingsSwitcher.style.color = "#AFB4FF";

    toggleRpc.style.setProperty('--toggle-color', "#AFB4FF");
    toggleBlur.style.setProperty('--toggle-color', "#AFB4FF");
    togglePreview.style.setProperty('--toggle-color', "#AFB4FF");
    toggleOled.style.setProperty('--toggle-color', "#AFB4FF");
    toggleRifeTta.style.setProperty('--toggle-color', "#AFB4FF");
    toggleRifeUhd.style.setProperty('--toggle-color', "#AFB4FF");
    togglefp16.style.setProperty('--toggle-color', "#AFB4FF");
    toggleShapes.style.setProperty('--toggle-color', "#AFB4FF");
    toggleTiling.style.setProperty('--toggle-color', "#AFB4FF");
    customModelCheck.style.setProperty('--toggle-color', "#AFB4FF");
    pythonCheck.style.setProperty('--toggle-color', "#AFB4FF");
    progressDone.style.setProperty('--bar-color', "#AFB4FF");
    scCheck.style.setProperty('--toggle-color', "#AFB4FF");
    skipCheck.style.setProperty('--toggle-color', "#AFB4FF");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#AFB4FF";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#AFB4FF");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#AFB4FF33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#AFB4FF";
    codecInterpolation.style.color = "#AFB4FF";
    codecUpscaling.style.color = "#AFB4FF";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#AFB4FF";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#AFB4FF";
        tooltips[i].style.setProperty('--tooltip-triangle', "#AFB4FF");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
lavender.addEventListener("click", lavenderSelect);

var rose = document.getElementById("theme-select-rose");

function roseSelect() {
    selector.style.top = "47%";
    selector.style.left = "73%";
    sessionStorage.setItem('theme', 'rose');
    settingsSaveButton.style.color = "#FF5D5D";
    interpolateButton.style.color = "#FF5D5D";
    upscalingButton.style.color = "#FF5D5D";
    restoreButton.style.color = "#FF5D5D";
    settingsSwitcher.style.color = "#FF5D5D";

    toggleRpc.style.setProperty('--toggle-color', "#FF5D5D");
    toggleBlur.style.setProperty('--toggle-color', "#FF5D5D");
    togglePreview.style.setProperty('--toggle-color', "#FF5D5D");
    toggleOled.style.setProperty('--toggle-color', "#FF5D5D");
    toggleRifeTta.style.setProperty('--toggle-color', "#FF5D5D");
    toggleRifeUhd.style.setProperty('--toggle-color', "#FF5D5D");
    togglefp16.style.setProperty('--toggle-color', "#FF5D5D");
    toggleShapes.style.setProperty('--toggle-color', "#FF5D5D");
    toggleTiling.style.setProperty('--toggle-color', "#FF5D5D");
    customModelCheck.style.setProperty('--toggle-color', "#FF5D5D");
    pythonCheck.style.setProperty('--toggle-color', "#FF5D5D");
    progressDone.style.setProperty('--bar-color', "#FF5D5D");
    scCheck.style.setProperty('--toggle-color', "#FF5D5D");
    skipCheck.style.setProperty('--toggle-color', "#FF5D5D");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#FF5D5D";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#FF5D5D");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#FF5D5D33";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#FF5D5D";
    codecInterpolation.style.color = "#FF5D5D";
    codecUpscaling.style.color = "#FF5D5D";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#FF5D5D";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#FF5D5D";
        tooltips[i].style.setProperty('--tooltip-triangle', "#FF5D5D");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
rose.addEventListener("click", roseSelect);

var grey = document.getElementById("theme-select-grey");

function greySelect() {
    selector.style.top = "47%";
    selector.style.left = "85%";
    sessionStorage.setItem('theme', 'grey');
    settingsSaveButton.style.color = "#696969";
    interpolateButton.style.color = "#696969";
    upscalingButton.style.color = "#696969";
    restoreButton.style.color = "#696969";
    settingsSwitcher.style.color = "#696969";

    toggleRpc.style.setProperty('--toggle-color', "#696969");
    toggleBlur.style.setProperty('--toggle-color', "#696969");
    togglePreview.style.setProperty('--toggle-color', "#696969");
    toggleOled.style.setProperty('--toggle-color', "#696969");
    toggleRifeTta.style.setProperty('--toggle-color', "#696969");
    toggleRifeUhd.style.setProperty('--toggle-color', "#696969");
    togglefp16.style.setProperty('--toggle-color', "#696969");
    toggleShapes.style.setProperty('--toggle-color', "#696969");
    toggleTiling.style.setProperty('--toggle-color', "#696969");
    customModelCheck.style.setProperty('--toggle-color', "#696969");
    pythonCheck.style.setProperty('--toggle-color', "#696969");
    progressDone.style.setProperty('--bar-color', "#696969");
    scCheck.style.setProperty('--toggle-color', "#696969");
    skipCheck.style.setProperty('--toggle-color', "#696969");

    for (var i = 0; i < queueProgressBar.length; i++) {
        queueProgressBar[i].style.background = "#696969";
    }

    for (var i = 0; i < side.length; i++) {
        side[i].style.setProperty('--side-hover', "#696969");
    }

    for (var i = 0; i < header.length; i++) {
        header[i].style.background = "#69696933";
    }

    var codecInterpolation = document.getElementById(sessionStorage.getItem("codecInterpolation"));
    var codecUpscaling = document.getElementById(sessionStorage.getItem("codecUpscaling") + "-up");
    var codecRestoration = document.getElementById(sessionStorage.getItem("codecRestoration") + "-res");
    codecRestoration.style.color = "#696969";
    codecInterpolation.style.color = "#696969";
    codecUpscaling.style.color = "#696969";
    for (var i = 0; i < mediaInfoText.length; i++) {
        mediaInfoText[i].style.color = "#696969";
    }
    for (var i = 0; i < tooltips.length; i++) {
        tooltips[i].style.backgroundColor = "#696969";
        tooltips[i].style.setProperty('--tooltip-triangle', "#696969");
    }
    sessionStorage.setItem('settingsSaved', 'false');
}
grey.addEventListener("click", greySelect);

var lightModeLayer = document.getElementById('light-mode');

if (process.platform == "win32") {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches == false) {
        console.log('light mode detected')
        lightModeLayer.style.visibility = 'visible';
    }
}

//load saved theme on startup
const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")
window.onload = function loadTheme() {
    // Read settings.json on launch
    fs.readFile(path.join(appDataPath, '/.enhancr/settings.json'), (err, settings) => {
        if (err) {
            theme = 'blue';
        };
        let json = JSON.parse(settings);
        // Set values
        var theme = json.settings[0].theme;

        switch (theme) {
            case 'blue':
                blueSelect();
                break;
            case 'pink':
                pinkSelect();
                break;
            case 'green':
                greenSelect();
                break;
            case 'purple':
                purpleSelect();
                break;
            case 'orange':
                orangeSelect();
                break;
            case 'yellow':
                yellowSelect();
                break;
            case 'red':
                redSelect();
                break;
            case 'sand':
                sandSelect();
                break;
            case 'mint':
                mintSelect();
                break;
            case 'salmon':
                salmonSelect();
                break;
            case 'egg':
                eggSelect();
                break;
            case 'lavender':
                lavenderSelect();
                break;
            case 'rose':
                roseSelect();
                break;
            case 'grey':
                greySelect();
                break;
            default:
                blueSelect();
        }
        sessionStorage.setItem('settingsSaved', 'true');
    });
}

class ThemeEngine {
    static getTheme(theme) {
        switch (theme) {
            case 'blue':
                return '#1e5cce';
            case 'pink':
                return '#ce1e6c'
            case 'green':
                return '#9ece1e';
            case 'purple':
                return '#601ece';
            case 'orange':
                return '#e36812';
            case 'yellow':
                return '#cece1e';
            case 'red':
                return '#ce2a1e';
            case 'sand':
                return '#E9DAC1';
            case 'mint':
                return '#8FE3CF';
            case 'salmon':
                return '#FFB3B3';
            case 'egg':
                return '#FFEF82';
            case 'lavender':
                return '#AFB4FF';
            case 'rose':
                return '#FF5D5D';
            case 'grey':
                return '#696969';
            default:
                return 'cornflowerblue';
        }
    }
}

module.exports = ThemeEngine;