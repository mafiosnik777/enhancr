const path = require("path");
const { ipcRenderer } = require("electron");
const fs = require("fs-extra");

const remote = require('@electron/remote');
const { BrowserWindow } = remote;

let win = BrowserWindow.getFocusedWindow();

const os = require('os');

var body = document.querySelector("body");

var videoInput = document.getElementById("input-video");
var openDialogInput = document.getElementById("dialog-video-input");
var videoInputText = document.getElementById("input-video-text");
var restoreVideoInputText = document.getElementById("restore-input-text");
var outputPath = document.getElementById("output-path");
var outputPathText = document.getElementById("output-path-text");

var upscaleEngineText = document.getElementById("upscale-engine-text");

document.addEventListener('keydown', function(e) {
    var handled = false;
    if (e.key !== undefined) {
        if((e.key == "Enter" && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    } else if (e.keyCode !== undefined) {
        if((e.keyCode == 13 && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    }
    if (handled) {
        let exportBtn = document.getElementById('export-btn');
        exportBtn.click()
        e.preventDefault();
    }
})


var x264Btn = document.getElementById("x264"),
    x265Btn = document.getElementById("x265"),
    AV1Btn = document.getElementById("AV1"),
    VP9Btn = document.getElementById("VP9"),
    ProResBtn = document.getElementById("ProRes"),
    LosslessBtn = document.getElementById("Lossless");

var hider = document.getElementById("hider");
var outputContainerInput = document.getElementById("output-container");
var containerDropdown = document.getElementById("container-dropdown");

var mp4Option = document.getElementById("mp4");
var mkvOption = document.getElementById("mkv");
var webmOption = document.getElementById("webm");
var movOption = document.getElementById("mov");
var framesOption = document.getElementById("frame-sequence");

var engineInput = document.getElementById("engine");
var engineDropdown = document.getElementById("engine-dropdown");

var cain = document.getElementById("cain");
var cainDml = document.getElementById("cain-dml");
var cainTrt = document.getElementById("cain-trt");

var modelInput = document.getElementById("model-selector");
var modelDropdown = document.getElementById("model-dropdown");

var rife23Option = document.getElementById("rife-23");
var rife4Option = document.getElementById("rife-4");
var rife46Option = document.getElementById("rife-46");

var mediaInfoContainer = document.getElementById("mediainfo-container");

var keyShortcut = document.getElementById("key-shortcut");

var winControls = document.getElementById("win-controls");

var videoInputText = document.getElementById("input-video-text");

var outputContainerSpan = document.getElementById("container-span");

var modelSpan = document.getElementById("model-span");

sessionStorage.setItem("currentTab", "interpolation");
sessionStorage.setItem("errorCount", "0");

const openBtn = document.querySelectorAll('[data-modal-target]')
const closeBtn = document.querySelectorAll("[data-modal-close]")
const overlay = document.getElementById("overlay")

var openProj = document.getElementById("new");

// set app version
document.getElementById('build-version').textContent = remote.app.getVersion()

// send ipc request from renderer to main process (open project)
function openProject() {
    ipcRenderer.send("open-project");
}
openProj.addEventListener("click", openProject);

// Handle event reply from main process
ipcRenderer.on("openproject", (event, openproject) => {
    sessionStorage.setItem("currentProject", openproject);
    window.location.replace("./app.html");
});

openBtn.forEach((btn) => {
    const modal = document.querySelector(btn.dataset.modalTarget)
    btn.addEventListener('click', (() => {
        openModal(modal)
    }))
})

closeBtn.forEach((btn) => {
    const modal = btn.closest(".modal")
    btn.addEventListener('click', (() => {
        closeModal(modal)
    }))
})

overlay.addEventListener('click', (() => {
    const modals = document.querySelectorAll('.modal.active')
    modals.forEach((modal) => {
        closeModal(modal)
    })
}))


function openModal(modal) {
    if (modal == undefined) return
    modal.classList.add('active')
    overlay.classList.add('active')
}

function closeModal(modal) {
    if (modal == undefined) return
    modal.classList.remove('active')
    overlay.classList.remove('active')
}

const blurLayer = document.getElementById('light-mode')
const border = document.getElementById('win10-border');

// change blur on win 11/10
let winOsBuild = parseInt(os.release().split(".")[2]);
if (winOsBuild >= 22000 && process.platform == 'win32') {
    blurLayer.style.visibility = 'hidden';
    border.style.visibility = 'hidden';
} else {
    blurLayer.style.visibility = 'visible';
    border.style.visibility = 'visible';
}

// light mode
if (!window.matchMedia('(prefers-color-scheme: dark)').matches) {
    console.log('light mode detected');
    blurLayer.style.visibility = 'visible';
}

const appDataPaths = {
    win32: process.env.APPDATA,
    darwin: `${process.env.HOME}/Library/Preferences`,
    linux: `${process.env.HOME}/.local/share`,
};
const appDataPath = path.resolve(appDataPaths[process.platform], '.enhancr');

if (!(localStorage.getItem('patreonUser') == null)) {
    // set users profile pic
    let profilePic = path.join(appDataPath, 'profile.jpeg');
    document.getElementById('patreon-user-img').src = profilePic;
} else {
    document.getElementById('patreon-user-img').style.visibility = 'hidden';
}

sessionStorage.setItem("multiInput", "false");

// clear out queue temp vars

let queueLength = Number(sessionStorage.getItem('queueLength'));

for(let i = 0; i < queueLength; i++) {
    try {
        sessionStorage.removeItem('queuePercent' + i);
        sessionStorage.removeItem('queueLength');
        sessionStorage.removeItem('percent');
        sessionStorage.remove('progress');
    } catch (error) {
    }
}

// drag and drop files
var dropOverlay = document.getElementById('overlay');
var dropSpan = document.getElementById('drop-span')
var dropIcon = document.getElementById('drop-icon')

let addtoQueueInterpolation = document.getElementById('interpolate-button-text');
let addtoQueueUpscaling = document.getElementById('upscaling-button-text');
let addtoQueueRestoration = document.getElementById('restore-button-text');

document.addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();

    let multipleFiles = [];
    sessionStorage.setItem("multiInput", "false");

    for (const item of event.dataTransfer.items) {
        let entry = item.webkitGetAsEntry()
        if (sessionStorage.getItem('currentTab') == 'interpolation') {
            if (event.dataTransfer.items.length > 1 && !entry.isDirectory) {
                let file = item.getAsFile();
                multipleFiles.push(file.path);
            } else if (entry.isFile) {
                let file = item.getAsFile();
                var videoInputPath = file.path;
                sessionStorage.setItem("inputPath", videoInputPath);
                if (
                    videoInputPath.length >= 55 &&
                    path.basename(videoInputPath).length >= 55
                ) {
                    videoInputText.textContent =
                        "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
                } else if (videoInputPath.length >= 55) {
                    videoInputText.textContent = "../" + path.basename(videoInputPath);
                } else {
                    videoInputText.textContent = videoInputPath;
                }
                addtoQueueInterpolation.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue`
                // autosave input in project file
                var currentProject = sessionStorage.getItem("currentProject");
                const data = JSON.parse(fs.readFileSync(currentProject));
                data.interpolation[0].inputFile = videoInputPath;
                fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
                console.log("Input path written to project file.");
            } else if (entry.isDirectory) {
                sessionStorage.setItem("multiInput", "true");
                let file = item.getAsFile();
                let dir = file.path;
                fs.readdir(dir, function (err, files) {
                    if (err) {
                        return console.log('Unable to scan directory: ' + err);
                    }
                    videoInputText.textContent = files.length + " files selected";
                    addtoQueueInterpolation.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${files.length})`
                    let multipleFileDir = [];
                    files.forEach(function (file) {
                        multipleFileDir.push(path.join(dir, file))
                    });
                    sessionStorage.setItem("interpolationMultiInput", JSON.stringify(multipleFileDir));
                });
            }
        } else if (sessionStorage.getItem('currentTab') == 'upscaling') {
            if (event.dataTransfer.items.length > 1 && !entry.isDirectory) {
                let file = item.getAsFile();
                multipleFiles.push(file.path);
            } else if (entry.isFile) {
                let file = item.getAsFile();
                var videoInputPath = file.path;
                sessionStorage.setItem("upscaleInputPath", videoInputPath);
                if (
                    videoInputPath.length >= 55 &&
                    path.basename(videoInputPath).length >= 55
                ) {
                    upscaleVideoInputText.textContent =
                        "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
                } else if (videoInputPath.length >= 55) {
                    upscaleVideoInputText.textContent = "../" + path.basename(videoInputPath);
                } else {
                    upscaleVideoInputText.textContent = videoInputPath;
                }
                addtoQueueUpscaling.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue`
                // autosave input in project file
                var currentProject = sessionStorage.getItem("currentProject");
                const data = JSON.parse(fs.readFileSync(currentProject));
                data.upscaling[0].inputFile = videoInputPath;
                fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
                console.log("Input path written to project file.");
            } else if (entry.isDirectory) {
                sessionStorage.setItem("multiInput", "true");
                let file = item.getAsFile();
                let dir = file.path;
                fs.readdir(dir, function (err, files) {
                    if (err) {
                        return console.log('Unable to scan directory: ' + err);
                    }
                    upscaleVideoInputText.textContent = files.length + " files selected";
                    addtoQueueUpscaling.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${files.length})`
                    let multipleFileDir = [];
                    files.forEach(function (file) {
                        multipleFileDir.push(path.join(dir, file))
                    });
                    sessionStorage.setItem("upscalingMultiInput", JSON.stringify(multipleFileDir));
                });
            }
        } else {
            if (event.dataTransfer.items.length > 1 && !entry.isDirectory) {
                let file = item.getAsFile();
                multipleFiles.push(file.path);
            } else if (entry.isFile) {
                let file = item.getAsFile();
                var videoInputPath = file.path;
                sessionStorage.setItem("inputPathRestore", videoInputPath);
                if (
                    videoInputPath.length >= 55 &&
                    path.basename(videoInputPath).length >= 55
                ) {
                    restoreVideoInputText.textContent =
                        "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
                } else if (videoInputPath.length >= 55) {
                    restoreVideoInputText.textContent = "../" + path.basename(videoInputPath);
                } else {
                    restoreVideoInputText.textContent = videoInputPath;
                }
                addtoQueueRestoration.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue`
                // autosave input in project file
                var currentProject = sessionStorage.getItem("currentProject");
                const data = JSON.parse(fs.readFileSync(currentProject));
                data.restoration[0].inputFile = videoInputPath;
                fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
                console.log("Input path written to project file.");
            } else if (entry.isDirectory) {
                sessionStorage.setItem("restorationMultiInput", "true");
                let file = item.getAsFile();
                let dir = file.path;
                fs.readdir(dir, function (err, files) {
                    if (err) {
                        return console.log('Unable to scan directory: ' + err);
                    }
                    restoreVideoInputText.textContent = files.length + " files selected";
                    addtoQueueRestoration.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${files.length})`
                    let multipleFileDir = [];
                    files.forEach(function (file) {
                        multipleFileDir.push(path.join(dir, file))
                    });
                    sessionStorage.setItem("restorationMultiInput", JSON.stringify(multipleFileDir));
                });
            }
        }
    }

    if (event.dataTransfer.items.length > 1 && multipleFiles.length != 0) {
        sessionStorage.setItem("multiInput", "true");
        if (sessionStorage.getItem('currentTab') == 'interpolation') {
            videoInputText.textContent = multipleFiles.length + " files selected";
            addtoQueueInterpolation.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${multipleFiles.length})`
            sessionStorage.setItem("interpolationMultiInput", JSON.stringify(multipleFiles));
        } else if (sessionStorage.getItem('currentTab') == 'upscaling') {
            upscaleVideoInputText.textContent = multipleFiles.length + " files selected";
            addtoQueueUpscaling.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${multipleFiles.length})`
            sessionStorage.setItem("upscalingMultiInput", JSON.stringify(multipleFiles));
        } else {
            restoreVideoInputText.textContent = multipleFiles.length + " files selected";
            addtoQueueRestoration.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Add to queue (${multipleFiles.length})`
            sessionStorage.setItem("restorationMultiInput", JSON.stringify(multipleFiles));
        }
        multipleFiles.forEach(file => {
            console.log(file);
        });
    }

    dropOverlay.style.opacity = 0;
    dropIcon.style.visibility = 'hidden';
    dropSpan.style.visibility = 'hidden';
});

document.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropOverlay.style.opacity = 1;
    dropIcon.style.visibility = 'visible';
    dropSpan.style.visibility = 'visible';
});

document.addEventListener('dragenter', (event) => { });

document.addEventListener('dragleave', (event) => {
    dropOverlay.style.opacity = 0;
    dropIcon.style.visibility = 'hidden';
    dropSpan.style.visibility = 'hidden';
});

//prevent dragging of text that locks up the UI
function preventDragging(elementId) {
    document.getElementById(elementId).addEventListener('dragstart', function(event) {
      event.preventDefault();
    });
}

preventDragging('ffmpeg-params');
preventDragging('ffmpeg-params-upscale');
preventDragging('ffmpeg-params-restoration');
preventDragging('denoise-strength');
preventDragging('shape-res');
preventDragging('tile-res');
preventDragging('num-streams');
preventDragging('sensitivity');

const interpolationSettingsBtn = document.getElementById('settings-btn-interpolation');
const modelsBtnUpscale = document.getElementById('models-btn-upscale');

// interpolation settings button
function changeToInterpolationSettings() {
    document.getElementById('settings-side').click();
    if (document.getElementById('settings-switcher').textContent == '<span>Page 1 / 2 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
        document.getElementById('settings-switcher').click();
    }
}
interpolationSettingsBtn.addEventListener('click', changeToInterpolationSettings);

// upscaling models button
function changeToModelsTab() {
    document.getElementById('models-side').click();
}
modelsBtnUpscale.addEventListener('click', changeToModelsTab);

const restorationSettings = document.getElementById('settings-btn-restoration');

// tensorrt settings button
function changeToTensorRTSettings() {
    document.getElementById('settings-side').click();
    if (document.getElementById('settings-switcher').innerHTML == '<span>Page 1 / 3 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
        document.getElementById('settings-switcher').click();
    }
    if (document.getElementById('settings-switcher').innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 3 / 3</span>') {
        document.getElementById('settings-switcher').click();
    }
}
restorationSettings.addEventListener('click', changeToTensorRTSettings);

// Linux specific code

if (process.platform == "linux") {
    body.style.backgroundColor = "#333333";
}

//Window controls
const isWin11 = os.release().split('.')[2] >= 22000;
const project = document.getElementById('project');
if (process.platform == "win32" && !isWin11 || process.platform == "linux") {
    winControls.style.visibility = "visible";
    project.style.marginLeft = '7%';
} else {
    winControls.style.visibility = "hidden";
    keyShortcut.textContent = "Ctrl + Enter";
}

// Window controls
const minimize = document.getElementById("minimize");
minimize.addEventListener("click", function () {
    ipcRenderer.send("minimize-window");
});
const close = document.getElementById("close");
close.addEventListener("click", function () {
    ipcRenderer.send("close-window");
});

// Interpolation
function toggleHider() {
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
}
// hide overlay when clicking out of dropdown
hider.addEventListener("click", toggleHider);


function toggleOutputDropdown() {
    hider.style.visibility = "visible";
    containerDropdown.style.visibility = "visible";
}
// toggle container dropdown
outputContainerInput.addEventListener("click", toggleOutputDropdown);

// change containers according to selection
mp4Option.addEventListener("click", function () {
    outputContainerSpan.textContent = ".mp4";
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    mediaInfoContainer.textContent = "mp4";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputContainer = ".mp4";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

mkvOption.addEventListener("click", function () {
    outputContainerSpan.textContent = ".mkv";
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    mediaInfoContainer.textContent = "mkv";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputContainer = ".mkv";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

movOption.addEventListener("click", function () {
    outputContainerSpan.textContent = ".mov";
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    mediaInfoContainer.textContent = "mov";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputContainer = ".mov";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

webmOption.addEventListener("click", function () {
    outputContainerSpan.textContent = ".webm";
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    mediaInfoContainer.textContent = "webm";
    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputContainer = ".webm";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

framesOption.addEventListener("click", function () {
    outputContainerSpan.textContent = "Frame Sequence";
    hider.style.visibility = "hidden";
    containerDropdown.style.visibility = "hidden";
    mediaInfoContainer.textContent = "Frame Sequence";
    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputContainer = "Frame Sequence";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

// toggle engine dropdown
engineInput.addEventListener("click", function () {
    hider.style.visibility = "visible";
    engineDropdown.style.visibility = "visible";
});

var interpolationEngineSpan = document.getElementById(
    "interpolation-engine-text"
);

var rvpv1Option = document.getElementById("rvp-v1");
var rvpv2Option = document.getElementById("rvp-v2");
var cvpv6Option = document.getElementById("cvp-v6");
var gmfssUnionOption = document.getElementById("gmf-union");
var gmfssFortunaOption = document.getElementById("gmf-fortuna");
var gmfssFortunaUnionOption = document.getElementById("gmf-fortuna-union");

// change engine (cain)
cain.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (NCNN)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
    gmfssUnionOption.style.display = 'none';
    gmfssFortunaOption.style.display = 'none';
    gmfssFortunaUnionOption.style.display = 'none';
    cvpv6Option.style.display = 'block';
    rvpv1Option.style.display = 'block';
    rvpv2Option.style.display = 'none';


    modelSpan.textContent = 'RVP - v1.0';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "cain";
    data.interpolation[0].model = "RVP - v1.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (cain-dml)
cainDml.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (DirectML)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
    gmfssUnionOption.style.display = 'none';
    gmfssFortunaOption.style.display = 'none';
    gmfssFortunaUnionOption.style.display = 'none';
    rvpv1Option.style.display = 'block';
    rvpv2Option.style.display = 'block';
    cvpv6Option.style.display = 'block';

    modelSpan.textContent = 'CVP - v6.0';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "cain-dml";
    data.interpolation[0].model = "CVP - v6.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (cain-trt)
cainTrt.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "Channel Attention - CAIN (TensorRT)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
    gmfssUnionOption.style.display = 'none';
    gmfssFortunaOption.style.display = 'none';
    gmfssFortunaUnionOption.style.display = 'none';
    rvpv1Option.style.display = 'block';
    rvpv2Option.style.display = 'block';
    cvpv6Option.style.display = 'block';

    modelSpan.textContent = 'CVP - v6.0';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "cain-trt";
    data.interpolation[0].model = "CVP - v6.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

var gmfss = document.getElementById("gmfss");

// change engine (gmfss)
gmfss.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "GMFlow - GMFSS (PyTorch)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
    gmfssUnionOption.style.display = 'block';
    gmfssFortunaOption.style.display = 'block';
    gmfssFortunaUnionOption.style.display = 'block';
    rvpv1Option.style.display = 'none';
    rvpv2Option.style.display = 'none';
    cvpv6Option.style.display = 'none';

    modelSpan.textContent = 'GMFSS - Fortuna';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "gmfss";
    data.interpolation[0].model = "GMFSS - Fortuna";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

var gmfssTrt = document.getElementById("gmfss-trt");

// change engine (gmfss-trt)
gmfssTrt.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "GMFlow - GMFSS (TensorRT)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'none';
    gmfssUnionOption.style.display = 'block';
    gmfssFortunaOption.style.display = 'block';
    gmfssFortunaUnionOption.style.display = 'block';
    rvpv1Option.style.display = 'none';
    rvpv2Option.style.display = 'none';
    cvpv6Option.style.display = 'none';

    modelSpan.textContent = 'GMFSS - Union';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "gmfss-trt";
    data.interpolation[0].model = "GMFSS - Union";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

var rife = document.getElementById("rife");
var rifeTrt = document.getElementById("rife-trt");

// change engine (rife)
rife.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "Optical Flow - RIFE (NCNN)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'block';
    rife4Option.style.display = 'block';
    rife46Option.style.display = 'block';
    gmfssUnionOption.style.display = 'none';
    gmfssFortunaOption.style.display = 'none';
    gmfssFortunaUnionOption.style.display = 'none';
    rvpv1Option.style.display = 'none';
    rvpv2Option.style.display = 'none';
    cvpv6Option.style.display = 'none';

    modelSpan.textContent = 'RIFE - v4.6';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "rife";
    data.interpolation[0].model = "RIFE - v4.6";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (rife)
rifeTrt.addEventListener("click", function () {
    interpolationEngineSpan.textContent = "Optical Flow - RIFE (TensorRT)";
    hider.style.visibility = "hidden";
    engineDropdown.style.visibility = "hidden";
    rife23Option.style.display = 'none';
    rife4Option.style.display = 'none';
    rife46Option.style.display = 'block';
    gmfssUnionOption.style.display = 'none';
    gmfssFortunaOption.style.display = 'none';
    gmfssFortunaUnionOption.style.display = 'none';
    rvpv1Option.style.display = 'none';
    rvpv2Option.style.display = 'none';
    cvpv6Option.style.display = 'none';

    modelSpan.textContent = 'RIFE - v4.6';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].engine = "rife-trt";
    data.interpolation[0].model = "RIFE - v4.6";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

//toggle model dropdown
modelInput.addEventListener("click", function () {
    hider.style.visibility = "visible";
    modelDropdown.style.visibility = "visible";
});

//change models according to selection
rife23Option.addEventListener("click", function () {
    modelSpan.textContent = "RIFE - v2.3";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "RIFE - v2.3";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

rife4Option.addEventListener("click", function () {
    modelSpan.textContent = "RIFE - v4.0";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "RIFE - v4.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

rife46Option.addEventListener("click", function () {
    modelSpan.textContent = "RIFE - v4.6";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "RIFE - v4.6";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

gmfssUnionOption.addEventListener("click", function () {
    modelSpan.textContent = "GMFSS - Union";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "GMFSS - Union";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

gmfssFortunaOption.addEventListener("click", function () {
    modelSpan.textContent = "GMFSS - Fortuna";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "GMFSS - Fortuna";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

gmfssFortunaUnionOption.addEventListener("click", function () {
    modelSpan.textContent = "GMFSS - Fortuna - Union";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "GMFSS - Fortuna - Union";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

rvpv1Option.addEventListener("click", function () {
    modelSpan.textContent = "RVP - v1.0";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "RVP - v1.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

rvpv2Option.addEventListener("click", function () {
    modelSpan.textContent = "RVP - v2.0";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "RVP - v2.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

cvpv6Option.addEventListener("click", function () {
    modelSpan.textContent = "CVP - v6.0";
    hider.style.visibility = "hidden";
    modelDropdown.style.visibility = "hidden";
    // autosave model in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].model = "CVP - v6.0";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Model written to project file.");
});

function uploadVideo() {
    openDialogInput.click();
}
// fires when input video dialog is clicked
videoInput.addEventListener("click", uploadVideo);

function processVideoInput() {
    var videoInputPath = openDialogInput.files[0].path;
    sessionStorage.setItem("inputPath", videoInputPath);

    // Stores rest of queue into sessionStorage
    Array.from(openDialogInput.files).forEach((file, num) => {
        sessionStorage.setItem("input" + num, file.path);
        num++;
    });

    if (
        videoInputPath.length >= 55 &&
        path.basename(videoInputPath).length >= 55
    ) {
        videoInputText.textContent =
            "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
    } else if (videoInputPath.length >= 55) {
        videoInputText.textContent = "../" + path.basename(videoInputPath);
    } else {
        videoInputText.textContent = videoInputPath;
    }
    // autosave input in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].inputFile = videoInputPath;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Input path written to project file.");
}
// fires when input video was selected by user
openDialogInput.addEventListener("change", processVideoInput);

function chooseOutDir() {
    ipcRenderer.send("file-request");
}
// fires when output path dialog is clicked
outputPath.addEventListener("click", chooseOutDir);

// Handle event reply from main process with selected output dir
ipcRenderer.on("file", (event, file) => {
    outputPathText.textContent = file;
    sessionStorage.setItem("outputPath", file);

    // autosave output path in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].outputPath = file;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output path written to project file.");
});

// Get ffmpeg parameters
var ffmpegParams = document.getElementById("ffmpeg-params");

// get theme

function getTheme() {
    if (sessionStorage.getItem('theme') === 'blue') {
        return '#1e5cce';
    }
    if (sessionStorage.getItem('theme') === 'pink') {
        return '#ce1e6c';
    }
    if (sessionStorage.getItem('theme') === 'green') {
        return '#9ece1e';
    }
    if (sessionStorage.getItem('theme') === 'purple') {
        return '#601ece';
    }
    if (sessionStorage.getItem('theme') === 'orange') {
        return '#e36812';
    }
    if (sessionStorage.getItem('theme') === 'yellow') {
        return '#fdfd06';
    }
    if (sessionStorage.getItem('theme') === 'red') {
        return '#ce2a1e';
    }
    if (sessionStorage.getItem('theme') === 'sand') {
        return '#E9DAC1';
    }
    if (sessionStorage.getItem('theme') === 'mint') {
        return '#8FE3CF';
    }
    if (sessionStorage.getItem('theme') === 'salmon') {
        return '#FFB3B3';
    }
    if (sessionStorage.getItem('theme') === 'egg') {
        return '#FFEF82';
    }
    if (sessionStorage.getItem('theme') === 'rose') {
        return '#FF5D5D';
    }
    if (sessionStorage.getItem('theme') === 'lavender') {
        return '#AFB4FF';
    }
    if (sessionStorage.getItem('theme') === 'grey') {
        return '#696969';
    }
}

async function setGPU() {
    let unsupportedCheck = document.getElementById('unsupported-check');
    const si = require('systeminformation');
    await si.graphics().then(data => sessionStorage.setItem('hasNVIDIA', data.controllers.some((controller) => controller.vendor.includes("NVIDIA"))));
    await si.graphics().then(data => sessionStorage.setItem('hasAMD', data.controllers.some((controller) => controller.vendor.includes("AMD"))));
    await si.graphics().then(data => sessionStorage.setItem('hasIntel', data.controllers.some((controller) => controller.vendor.includes("Intel"))));
    // check for gpus and set final gpu based on hierarchy
    if (sessionStorage.getItem('hasIntel') == "true" && !unsupportedCheck.checked) {
        sessionStorage.setItem('gpu', 'Intel');
        document.getElementById('cain-trt').style.display = 'none';
        document.getElementById('rife-trt').style.display = 'none';
        document.getElementById('realesrgan-tensorrt').style.display = 'none';
        document.getElementById('realcugan-tensorrt').style.display = 'none';
        document.getElementById('dpir').style.display = 'none';
        document.getElementById('anime-video').style.display = 'none';
        document.getElementById('rife').click();
        document.getElementById('realesrgan-ncnn').click();
        document.getElementById('anime-video-ncnn').click();
    }
    if (sessionStorage.getItem('hasAMD') == "true" && !unsupportedCheck.checked) {
        sessionStorage.setItem('gpu', 'AMD');
        document.getElementById('cain-trt').style.display = 'none';
        document.getElementById('rife-trt').style.display = 'none';
        document.getElementById('realesrgan-tensorrt').style.display = 'none';
        document.getElementById('realcugan-tensorrt').style.display = 'none';
        document.getElementById('dpir').style.display = 'none';
        document.getElementById('anime-video').style.display = 'none';
        document.getElementById('rife').click();
        document.getElementById('realesrgan-ncnn').click();
        document.getElementById('anime-video-ncnn').click();
    }
    if (sessionStorage.getItem('hasNVIDIA') == "true") {
        sessionStorage.setItem('gpu', 'NVIDIA')
        document.getElementById('cain-trt').style.display = 'block';
        document.getElementById('rife-trt').style.display = 'block';
        document.getElementById('realesrgan-tensorrt').style.display = 'block';
        document.getElementById('realcugan-tensorrt').style.display = 'block';
        document.getElementById('dpir').style.display = 'block';
        document.getElementById('anime-video').style.display = 'block';
    }
}
setGPU();

function supportsCodec(codec) {
    if (codec == 'H264') {
        if (document.getElementById('H264-hw').innerHTML == '<i class="fa-solid fa-check"></i> H264') {
            return true;
        } else if (document.getElementById('H264-hw').innerHTML == '<i class="fa-solid fa-question"></i> H264') {
            return true;
        } else {
            return false;
        }
    }
    if (codec == 'H265') {
        if (document.getElementById('H265-hw').innerHTML == '<i class="fa-solid fa-check"></i> H265') {
            return true;
        } else if (document.getElementById('H265-hw').innerHTML == '<i class="fa-solid fa-question"></i> H265') {
            return true;
        } else {
            return false;
        }
    }
    if (codec == 'AV1') {
        if (document.getElementById('AV1-hw').innerHTML == '<i class="fa-solid fa-check"></i> AV1') {
            return true;
        } else if (document.getElementById('AV1-hw').innerHTML == '<i class="fa-solid fa-question"></i> AV1') {
            return true;
        } else {
            return false;
        }
    }

}

function getHWEncoder(gpu, codec) {
    if (gpu == "NVIDIA") {
        if (codec == "H264" && supportsCodec('H264')) {
            return "json.codecs[0].h264_nvenc";
        }
        if (codec == "H265" && supportsCodec('H265')) {
            return "json.codecs[0].hevc_nvenc";
        }
        if (codec == "AV1" && supportsCodec('AV1')) {
            return "json.codecs[0].av1_nvenc";
        }
    } else if (gpu == "AMD") {
        if (codec == "H264" && supportsCodec('H264')) {
            return "json.codecs[0].h264_amf";
        }
        if (codec == "H265" && supportsCodec('H265')) {
            return "json.codecs[0].hevc_amf";
        }
        if (codec == "AV1" && supportsCodec('AV1')) {
            return "json.codecs[0].av1_amf";
        }
    } else if (gpu == "Intel") {
        if (codec == "H264" && supportsCodec('H264')) {
            return "json.codecs[0].h264_qsv";
        }
        if (codec == "H265" && supportsCodec('H265')) {
            return "json.codecs[0].hevc_qsv";
        }
        if (codec == "AV1" && supportsCodec('AV1')) {
            return "json.codecs[0].av1_qsv";
        }
    }
}

function changeCodecH264() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    console.log(gpu);
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H264"));
    let codec = hwEncode ? hwParams || json.codecs[0].x264 : json.codecs[0].x264;
    ffmpegParams.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecInterpolation', 'x264');

    x265Btn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    LosslessBtn.style.color = "rgb(208, 208, 208)";
    x264Btn.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "H264";
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecH265() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H265"));
    let codec = hwEncode ? hwParams || json.codecs[0].x265 : json.codecs[0].x265;
    ffmpegParams.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecInterpolation', 'x265');

    x264Btn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    LosslessBtn.style.color = "rgb(208, 208, 208)";
    x265Btn.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "H265";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecAV1() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "AV1"));
    let codec = hwEncode ? hwParams || json.codecs[0].AV1 : json.codecs[0].AV1;
    ffmpegParams.value = codec;

    sessionStorage.setItem('codecInterpolation', 'AV1');

    var theme = getTheme();

    x264Btn.style.color = "rgb(208, 208, 208)";
    x265Btn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    LosslessBtn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "AV1";
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecVP9() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParams.value = json.codecs[0].VP9;

    var theme = getTheme();

    sessionStorage.setItem('codecInterpolation', 'VP9');

    x264Btn.style.color = "rgb(208, 208, 208)";
    x265Btn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    LosslessBtn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "VP9";
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecProRes() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParams.value = json.codecs[0].ProRes;

    sessionStorage.setItem('codecInterpolation', 'ProRes');

    var theme = getTheme();

    x264Btn.style.color = "rgb(208, 208, 208)";
    x265Btn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "ProRes";
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecLossless() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParams.value = json.codecs[0].Lossless;

    var theme = getTheme();

    sessionStorage.setItem('codecInterpolation', 'Lossless');

    x264Btn.style.color = "rgb(208, 208, 208)";
    x265Btn.style.color = "rgb(208, 208, 208)";
    AV1Btn.style.color = "rgb(208, 208, 208)";
    ProResBtn.style.color = "rgb(208, 208, 208)";
    VP9Btn.style.color = "rgb(208, 208, 208)";
    LosslessBtn.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].codec = "Lossless";
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

// Codec event listeners
x264Btn.addEventListener("click", changeCodecH264);
x265Btn.addEventListener("click", changeCodecH265);
AV1Btn.addEventListener("click", changeCodecAV1);
VP9Btn.addEventListener("click", changeCodecVP9);
ProResBtn.addEventListener("click", changeCodecProRes);
LosslessBtn.addEventListener("click", changeCodecLossless);

var scaleInput = document.getElementById("factor-selector"),
    scaleDropdown = document.getElementById("factor-dropdown"),
    scaleSpan = document.getElementById("factor-span"),
    scale2Option = document.getElementById("factor2"),
    scale3Option = document.getElementById("factor3"),
    scale4Option = document.getElementById("factor4");

var x265BtnUp = document.getElementById("x265-up"),
    AV1BtnUp = document.getElementById("AV1-up"),
    VP9BtnUp = document.getElementById("VP9-up"),
    ProResBtnUp = document.getElementById("ProRes-up"),
    LosslessBtnUp = document.getElementById("Lossless-up"),
    x264BtnUp = document.getElementById("x264-up");

var engineInputUpscale = document.getElementById("engine-upscale");
var engineDropdownUpscale = document.getElementById("engine-dropdown-upscale");
var shuffleCuganOption = document.getElementById("shufflecugan");
var shuffleCuganNcnnOption = document.getElementById("shufflecugan-ncnn");
var realesrganOption = document.getElementById("realesrgan-tensorrt");
var realcuganOption = document.getElementById("realcugan-tensorrt");
var swinIROption = document.getElementById("swinir");
var realesrganNcnnOption = document.getElementById("realesrgan-ncnn");
var realesrganDmlOption = document.getElementById("realesrgan-dml");

var upscaleOutputPath = document.getElementById("upscale-output-path");
var upscaleOutputPathText = document.getElementById("upscale-output-path-text");

var openDialogInputUpscale = document.getElementById("dialog-upscale-input");
var videoInputUpscale = document.getElementById("upscale-input-video");
var upscaleVideoInputText = document.getElementById("upscale-input-text");

// Upscaling
function uploadVideoUp() {
    openDialogInputUpscale.click();
}
// fires when input video dialog is clicked
videoInputUpscale.addEventListener("click", uploadVideoUp);

function processVideoInputUp() {
    var videoInputPath = openDialogInputUpscale.files[0].path;
    sessionStorage.setItem("upscaleInputPath", videoInputPath);
    if (
        videoInputPath.length >= 55 &&
        path.basename(videoInputPath).length >= 55
    ) {
        upscaleVideoInputText.textContent =
            "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
    } else if (videoInputPath.length >= 55) {
        upscaleVideoInputText.textContent = "../" + path.basename(videoInputPath);
    } else {
        upscaleVideoInputText.textContent = videoInputPath;
    }
    // autosave input in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].inputFile = videoInputPath;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Input path written to project file.");
}
// fires when input video was selected by user
openDialogInputUpscale.addEventListener("change", processVideoInputUp);

// output path
function chooseOutDirUp() {
    ipcRenderer.send("file-request-up");
}
// fires when output path dialog is clicked
upscaleOutputPath.addEventListener("click", chooseOutDirUp);

// Handle event reply from main process with selected output dir
ipcRenderer.on("file-up", (event, file) => {
    upscaleOutputPathText.textContent = file;
    sessionStorage.setItem("upscaleOutputPath", file);

    // autosave output path in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputPath = file;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output path written to project file.");
});

// Get ffmpeg parameters
var ffmpegParamsUpscale = document.getElementById("ffmpeg-params-upscale");

function changeCodecH264Upscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H264"));
    let codec = hwEncode ? hwParams || json.codecs[0].x264 : json.codecs[0].x264;
    ffmpegParamsUpscale.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'x264');

    x265BtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    LosslessBtnUp.style.color = "rgb(208, 208, 208)";
    x264BtnUp.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "H264";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecH265Upscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H265"));
    let codec = hwEncode ? hwParams || json.codecs[0].x265 : json.codecs[0].x265;
    ffmpegParamsUpscale.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'x265');

    x264BtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    LosslessBtnUp.style.color = "rgb(208, 208, 208)";
    x265BtnUp.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "H265";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecAV1Upscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "AV1"));
    let codec = hwEncode ? hwParams || json.codecs[0].AV1 : json.codecs[0].AV1;
    ffmpegParamsUpscale.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'AV1');

    x264BtnUp.style.color = "rgb(208, 208, 208)";
    x265BtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    LosslessBtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "AV1";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecVP9Upscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsUpscale.value = json.codecs[0].VP9;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'VP9');

    x264BtnUp.style.color = "rgb(208, 208, 208)";
    x265BtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    LosslessBtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "VP9";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecProResUpscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsUpscale.value = json.codecs[0].ProRes;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'ProRes');

    x264BtnUp.style.color = "rgb(208, 208, 208)";
    x265BtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "ProRes";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecLosslessUpscale() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsUpscale.value = json.codecs[0].Lossless;

    var theme = getTheme();

    sessionStorage.setItem('codecUpscaling', 'Lossless');

    x264BtnUp.style.color = "rgb(208, 208, 208)";
    x265BtnUp.style.color = "rgb(208, 208, 208)";
    AV1BtnUp.style.color = "rgb(208, 208, 208)";
    ProResBtnUp.style.color = "rgb(208, 208, 208)";
    VP9BtnUp.style.color = "rgb(208, 208, 208)";
    LosslessBtnUp.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].codec = "Lossless";
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

x264BtnUp.addEventListener("click", changeCodecH264Upscale);
x265BtnUp.addEventListener("click", changeCodecH265Upscale);
AV1BtnUp.addEventListener("click", changeCodecAV1Upscale);
VP9BtnUp.addEventListener("click", changeCodecVP9Upscale);
ProResBtnUp.addEventListener("click", changeCodecProResUpscale);
LosslessBtnUp.addEventListener("click", changeCodecLosslessUpscale);

var hiderUpscale = document.getElementById("hider-upscale"),
    containerDropdownUpscale = document.getElementById("container-dropdown-upscale");

var outputContainerInputUpscale = document.getElementById("output-container-upscale"),
    mp4OptionUp = document.getElementById("mp4-up"),
    mkvOptionUp = document.getElementById("mkv-up"),
    movOptionUp = document.getElementById("mov-up"),
    webmOptionUp = document.getElementById("webm-up"),
    framesOptionUp = document.getElementById("frame-sequence-up");
var outputContainerSpanUp = document.getElementById("container-span-up");

function toggleHiderUpscale() {
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    scaleDropdown.style.visibility = "hidden";
}
// hide overlay when clicking out of dropdown
hiderUpscale.addEventListener("click", toggleHiderUpscale);

function toggleOutputDropdownUpscale() {
    hiderUpscale.style.visibility = "visible";
    containerDropdownUpscale.style.visibility = "visible";
}
// toggle container dropdown
outputContainerInputUpscale.addEventListener(
    "click",
    toggleOutputDropdownUpscale
);

var mediaInfoContainerUp = document.getElementById("mediainfo-containerUp");
var outputContainerSpanUp = document.getElementById("container-span-up");

// change containers according to selection
mp4OptionUp.addEventListener("click", function () {
    outputContainerSpanUp.textContent = ".mp4";
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    mediaInfoContainerUp.textContent = "mp4";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputContainer = ".mp4";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

mkvOptionUp.addEventListener("click", function () {
    outputContainerSpanUp.textContent = ".mkv";
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    mediaInfoContainerUp.textContent = "mkv";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputContainer = ".mkv";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

movOptionUp.addEventListener("click", function () {
    outputContainerSpanUp.textContent = ".mov";
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    mediaInfoContainerUp.textContent = "mov";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputContainer = ".mov";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

webmOptionUp.addEventListener("click", function () {
    outputContainerSpanUp.textContent = ".webm";
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    mediaInfoContainerUp.textContent = "webm";
    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputContainer = ".webm";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

framesOptionUp.addEventListener("click", function () {
    outputContainerSpanUp.textContent = "Frame Sequence";
    hiderUpscale.style.visibility = "hidden";
    containerDropdownUpscale.style.visibility = "hidden";
    mediaInfoContainerUp.textContent = "Frame Sequence";
    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].outputContainer = "Frame Sequence";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

var factorSpan = document.getElementById("factor-span");
var factor2 = document.getElementById("factor2");
var factor3 = document.getElementById("factor3");
var factor4 = document.getElementById("factor4");

// toggle engine dropdown
engineInputUpscale.addEventListener("click", function () {
    hiderUpscale.style.visibility = "visible";
    engineDropdownUpscale.style.visibility = "visible";
});

// change engine (shufflecugan)
shuffleCuganOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - ShuffleCUGAN (TensorRT)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factorSpan.textContent = "2x";
    factor2.style.display = "block";
    factor3.style.display = "none";
    factor4.style.display = "none";
    sessionStorage.setItem("upscaleFactor", "2");

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "shufflecugan";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (shufflecugan)
shuffleCuganNcnnOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - ShuffleCUGAN (NCNN)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factorSpan.textContent = "2x";
    factor2.style.display = "block";
    factor3.style.display = "none";
    factor4.style.display = "none";
    sessionStorage.setItem("upscaleFactor", "2");

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "shufflecugan-ncnn";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-ncnn)
realesrganNcnnOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - RealESRGAN (NCNN)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factor2.style.display = "block";
    factor3.style.display = "block";
    factor4.style.display = "block";

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "realesrgan-ncnn";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-dml)
realesrganDmlOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - RealESRGAN (DirectML)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factor2.style.display = "block";
    factor3.style.display = "block";
    factor4.style.display = "block";

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "realesrgan-dml";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-trt)
realesrganOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - RealESRGAN (TensorRT)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factor2.style.display = "block";
    factor3.style.display = "block";
    factor4.style.display = "block";

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "realesrgan";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realcugan-trt)
realcuganOption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - RealCUGAN (TensorRT)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factor2.style.display = "block";
    factor3.style.display = "block";
    factor4.style.display = "none";

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "realcugan";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (swinir)
swinIROption.addEventListener("click", function () {
    upscaleEngineText.textContent = "Upscaling - SwinIR (TensorRT)";
    hiderUpscale.style.visibility = "hidden";
    engineDropdownUpscale.style.visibility = "hidden";
    factor2.style.display = "block";
    factor3.style.display = "none";
    factor4.style.display = "none";

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].engine = "swinir";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

//toggle scale dropdown
scaleInput.addEventListener("click", function () {
    hiderUpscale.style.visibility = "visible";
    scaleDropdown.style.visibility = "visible";
});

//change scale according to selection
scale2Option.addEventListener("click", function () {
    scaleSpan.textContent = "2x";
    hiderUpscale.style.visibility = "hidden";
    scaleDropdown.style.visibility = "hidden";
    sessionStorage.setItem("upscaleFactor", "2");
    // autosave scale in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].scale = "2x";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Scale written to project file.");
});

scale3Option.addEventListener("click", function () {
    scaleSpan.textContent = "3x";
    hiderUpscale.style.visibility = "hidden";
    scaleDropdown.style.visibility = "hidden";
    sessionStorage.setItem("upscaleFactor", "3");
    // autosave scale in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].scale = "3x";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Scale written to project file.");
});
scale4Option.addEventListener("click", function () {
    scaleSpan.textContent = "4x";
    hiderUpscale.style.visibility = "hidden";
    scaleDropdown.style.visibility = "hidden";
    sessionStorage.setItem("upscaleFactor", "4");
    // autosave scale in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].scale = "4x";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Scale written to project file.");
});

// Restoration

const openDialogInputRes = document.getElementById('dialog-restore-input');
const videoInputRes = document.getElementById('input-video-restoration');
const restoreInputText = document.getElementById('restore-input-text');

function uploadVideoRes() {
    openDialogInputRes.click();
}
// fires when input video dialog is clicked
videoInputRes.addEventListener("click", uploadVideoRes);

function processVideoInputRes() {
    var videoInputPathRes = openDialogInputRes.files[0].path;
    sessionStorage.setItem("inputPathRestore", videoInputPathRes);

    // Stores rest of queue into sessionStorage
    Array.from(openDialogInputRes.files).forEach((file, num) => {
        sessionStorage.setItem("input" + num, file.path);
        num++;
    });

    if (
        videoInputPathRes.length >= 55 &&
        path.basename(videoInputPathRes).length >= 55
    ) {
        restoreInputText.textContent =
            "../" + path.basename(videoInputPathRes).substr(0, 55) + "\u2026";
    } else if (videoInputPathRes.length >= 55) {
        restoreInputText.textContent = "../" + path.basename(videoInputPathRes);
    } else {
        restoreInputText.textContent = videoInputPathRes;
    }
    // autosave input in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].inputFile = videoInputPathRes;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Input path written to project file.");
}
// fires when input video was selected by user
openDialogInputRes.addEventListener("change", processVideoInputRes);

var outputPathRes = document.getElementById("output-path-restoration");
var outputPathTextRes = document.getElementById("restore-output-path-text");

function chooseOutDirRes() {
    ipcRenderer.send("file-request-res");
}
// fires when output path dialog is clicked
outputPathRes.addEventListener("click", chooseOutDirRes);

// Handle event reply from main process with selected output dir
ipcRenderer.on("file-res", (event, file) => {
    outputPathTextRes.textContent = file;
    sessionStorage.setItem("outputPathRestoration", file);

    // autosave output path in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputPath = file;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output path written to project file.");
});

var x265BtnRes = document.getElementById("x265-res"),
    AV1BtnRes = document.getElementById("AV1-res"),
    VP9BtnRes = document.getElementById("VP9-res"),
    ProResBtnRes = document.getElementById("ProRes-res"),
    LosslessBtnRes = document.getElementById("Lossless-res"),
    x264BtnRes = document.getElementById("x264-res");

// Get ffmpeg parameters
var ffmpegParamsRestoration = document.getElementById("ffmpeg-params-restoration");

function changeCodecH264Restoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H264"));
    let codec = hwEncode ? hwParams || json.codecs[0].x264 : json.codecs[0].x264;
    ffmpegParamsRestoration.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'x264');

    x265BtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    LosslessBtnRes.style.color = "rgb(208, 208, 208)";
    x264BtnRes.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "H264";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecH265Restoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "H265"));
    let codec = hwEncode ? hwParams || json.codecs[0].x265 : json.codecs[0].x265;
    ffmpegParamsRestoration.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'x265');

    x264BtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    LosslessBtnRes.style.color = "rgb(208, 208, 208)";
    x265BtnRes.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "H265";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecAV1Restoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }
    let gpu = sessionStorage.getItem('gpu');
    let hwEncode = document.getElementById('hwencode-check').checked;
    let hwParams = eval(getHWEncoder(gpu, "AV1"));
    let codec = hwEncode ? hwParams || json.codecs[0].AV1 : json.codecs[0].AV1;
    ffmpegParamsRestoration.value = codec;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'AV1');

    x264BtnRes.style.color = "rgb(208, 208, 208)";
    x265BtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    LosslessBtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "AV1";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecVP9Restoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsRestoration.value = json.codecs[0].VP9;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'VP9');

    x264BtnRes.style.color = "rgb(208, 208, 208)";
    x265BtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    LosslessBtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = theme;

    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "VP9";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecProResRestoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsRestoration.value = json.codecs[0].ProRes;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'ProRes');

    x264BtnRes.style.color = "rgb(208, 208, 208)";
    x265BtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "ProRes";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

function changeCodecLosslessRestoration() {
    try {
        const jsonString = fs.readFileSync(path.join(__dirname, '..', "/src/codecs.json"));
        var json = JSON.parse(jsonString);
    } catch (err) {
        console.log(err);
        return;
    }

    ffmpegParamsRestoration.value = json.codecs[0].Lossless;

    var theme = getTheme();

    sessionStorage.setItem('codecRestoration', 'Lossless');

    x264BtnRes.style.color = "rgb(208, 208, 208)";
    x265BtnRes.style.color = "rgb(208, 208, 208)";
    AV1BtnRes.style.color = "rgb(208, 208, 208)";
    ProResBtnRes.style.color = "rgb(208, 208, 208)";
    VP9BtnRes.style.color = "rgb(208, 208, 208)";
    LosslessBtnRes.style.color = theme;
    // autosave codec in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].codec = "Lossless";
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Codec written to project file.");
}

x264BtnRes.addEventListener("click", changeCodecH264Restoration);
x265BtnRes.addEventListener("click", changeCodecH265Restoration);
AV1BtnRes.addEventListener("click", changeCodecAV1Restoration);
VP9BtnRes.addEventListener("click", changeCodecVP9Restoration);
ProResBtnRes.addEventListener("click", changeCodecProResRestoration);
LosslessBtnRes.addEventListener("click", changeCodecLosslessRestoration);

// set codec on startup if hardware encode is checked
function setCodecs() {
    var waitforGPUInitialization = setInterval(() => {
        if (sessionStorage.getItem('gpu') != null) {
            let interpolationCodec = sessionStorage.getItem('codecInterpolation');
            let upscalingCodec = sessionStorage.getItem('codecUpscaling');
            let restorationCodec = sessionStorage.getItem('codecRestoration');

            let codecs = [interpolationCodec, upscalingCodec, restorationCodec];
            for (let index = 0; index < codecs.length; index++) {
                let codec = codecs[index];
                switch (index) {
                    case 0:
                        switch (codec) {
                            case 'x264':
                                changeCodecH264();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'x265':
                                changeCodecH265();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'AV1':
                                changeCodecAV1();
                                clearInterval(waitforGPUInitialization);
                                break;
                        }
                        break;
                    case 1:
                        switch (codec) {
                            case 'x264':
                                changeCodecH264Upscale();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'x265':
                                changeCodecH265Upscale();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'AV1':
                                changeCodecAV1Upscale();
                                clearInterval(waitforGPUInitialization);
                                break;
                        }
                        break;
                    case 2:
                        switch (codec) {
                            case 'x264':
                                changeCodecH264Restoration();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'x265':
                                changeCodecH265Restoration();
                                clearInterval(waitforGPUInitialization);
                                break;
                            case 'AV1':
                                changeCodecAV1Restoration();
                                clearInterval(waitforGPUInitialization);
                                break;
                        }
                        break;
                }
            }
        }
    }, 1000)
};
setCodecs();

const hwEncodeCheck = document.getElementById('hwencode-check');

hwEncodeCheck.addEventListener('click', () => {
    if (hwEncodeCheck.checked) setCodecs();
})

var hiderRestore = document.getElementById("hider-restore"),
    containerDropdownRestoration = document.getElementById("container-dropdown-restoration");

var outputContainerInputRestoration = document.getElementById("output-container-restoration"),
    mp4OptionRes = document.getElementById("mp4-res"),
    mkvOptionRes = document.getElementById("mkv-res"),
    movOptionRes = document.getElementById("mov-res"),
    webmOptionRes = document.getElementById("webm-res"),
    framesOptionRes = document.getElementById("frame-sequence-res");
var outputContainerSpanRes = document.getElementById("container-span-res");

const modelDropdownRes = document.getElementById('model-dropdown-restoration')

function toggleHiderRestore() {
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelDropdownRes.style.visibility = "hidden";
}
// hide overlay when clicking out of dropdown
hiderRestore.addEventListener("click", toggleHiderRestore);

function toggleOutputDropdownRestoration() {
    hiderRestore.style.visibility = "visible";
    containerDropdownRestoration.style.visibility = "visible";
}
// toggle container dropdown
outputContainerInputRestoration.addEventListener("click", toggleOutputDropdownRestoration);

var mediaInfoContainerRes = document.getElementById("mediainfo-containerRes");
var outputContainerSpanRes = document.getElementById("container-span-res");

// change containers according to selection
mp4OptionRes.addEventListener("click", function () {
    outputContainerSpanRes.textContent = ".mp4";
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    mediaInfoContainerRes.textContent = "mp4";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputContainer = ".mp4";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

mkvOptionRes.addEventListener("click", function () {
    outputContainerSpanRes.textContent = ".mkv";
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    mediaInfoContainerRes.textContent = "mkv";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputContainer = ".mkv";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

movOptionRes.addEventListener("click", function () {
    outputContainerSpanRes.textContent = ".mov";
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    mediaInfoContainerRes.textContent = "mov";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputContainer = ".mov";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

webmOptionRes.addEventListener("click", function () {
    outputContainerSpanUp.textContent = ".webm";
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    mediaInfoContainerRes.textContent = "webm";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputContainer = ".webm";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

framesOptionRes.addEventListener("click", function () {
    outputContainerSpanUp.textContent = "Frame Sequence";
    hiderRestore.style.visibility = "hidden";
    containerDropdownRestoration.style.visibility = "hidden";
    mediaInfoContainerRes.textContent = "Frame Sequence";

    // autosave container in project file
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].outputContainer = "Frame Sequence";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Output container written to project file.");
});

const engineInputRestore = document.getElementById('engine-restoration');
const engineDropdownRestore = document.getElementById('engine-dropdown-restoration');


// toggle engine dropdown
engineInputRestore.addEventListener("click", function () {
    hiderRestore.style.visibility = "visible";
    engineDropdownRestore.style.visibility = "visible";
});

const dpirOption = document.getElementById('dpir');
const dpirDmlOption = document.getElementById('dpir-dml');
const scunetOption = document.getElementById('scunet');
const esrgan1xOption = document.getElementById('anime-video');
const esrgan1xNcnnOption = document.getElementById('anime-video-ncnn');
const esrgan1xDmlOption = document.getElementById('anime-video-dml');
const restoreEngineText = document.getElementById('restoration-engine-text');
const modelSpanRes = document.getElementById('model-span-restoration');
const denoiseSharpen = document.getElementById('denoise-sharpen');
const denoise = document.getElementById('denoise');
const deblock = document.getElementById('deblock');
const strength15 = document.getElementById('strength-15');
const strength25 = document.getElementById('strength-25');
const strength50 = document.getElementById('strength-50');
const strengthGAN = document.getElementById('strength-gan');

// change engine (scunet)
scunetOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - ScuNET (TensorRT)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Strength: 50%";
    denoiseSharpen.style.display = 'none';
    denoise.style.display = 'none';
    deblock.style.display = 'none';
    strength15.style.display = 'block';
    strength25.style.display = 'block';
    strength50.style.display = 'block';
    strengthGAN.style.display = 'block';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "scunet";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (dpir)
dpirOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - DPIR (TensorRT)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Denoise";
    denoiseSharpen.style.display = 'none';
    denoise.style.display = 'block';
    deblock.style.display = 'block';
    strength15.style.display = 'none';
    strength25.style.display = 'none';
    strength50.style.display = 'none';
    strengthGAN.style.display = 'none';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "dpir";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (dpir-dml)
dpirDmlOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - DPIR (DirectML)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Denoise";
    denoiseSharpen.style.display = 'none';
    denoise.style.display = 'block';
    deblock.style.display = 'block';
    strength15.style.display = 'none';
    strength25.style.display = 'none';
    strength50.style.display = 'none';
    strengthGAN.style.display = 'none';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "dpir-dml";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-1x-trt)
esrgan1xOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - RealESRGAN (1x) (TensorRT)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Denoise/Sharpen";
    denoiseSharpen.style.display = 'block';
    denoise.style.display = 'none';
    deblock.style.display = 'none';
    strength15.style.display = 'none';
    strength25.style.display = 'none';
    strength50.style.display = 'none';
    strengthGAN.style.display = 'none';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "esrgan-1x-trt";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-1x-ncnn)
esrgan1xNcnnOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - RealESRGAN (1x) (NCNN)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Denoise/Sharpen";
    denoiseSharpen.style.display = 'block';
    denoise.style.display = 'none';
    deblock.style.display = 'none';
    strength15.style.display = 'none';
    strength25.style.display = 'none';
    strength50.style.display = 'none';
    strengthGAN.style.display = 'none';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "esrgan-1x-ncnn";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

// change engine (realesrgan-1x-dml)
esrgan1xDmlOption.addEventListener("click", function () {
    restoreEngineText.textContent = "Restoration - RealESRGAN (1x) (DirectML)";
    hiderRestore.style.visibility = "hidden";
    engineDropdownRestore.style.visibility = "hidden";
    modelSpanRes.textContent = "Denoise/Sharpen";
    denoiseSharpen.style.display = 'block';
    denoise.style.display = 'none';
    deblock.style.display = 'none';
    strength15.style.display = 'none';
    strength25.style.display = 'none';
    strength50.style.display = 'none';
    strengthGAN.style.display = 'none';

    // autosave
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].engine = "esrgan-1x-dml";
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Engine written to project file.");
});

const modelInputRes = document.getElementById('model-selector-restoration');

denoiseSharpen.addEventListener("click", function () {
    modelSpanRes.textContent = "Denoise/Sharpen";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});

denoise.addEventListener("click", function () {
    modelSpanRes.textContent = "Denoise";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});

deblock.addEventListener("click", function () {
    modelSpanRes.textContent = "Deblock";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});

strength15.addEventListener("click", function () {
    modelSpanRes.textContent = "Strength: 15%";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});
strength25.addEventListener("click", function () {
    modelSpanRes.textContent = "Strength: 25%";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});
strength50.addEventListener("click", function () {
    modelSpanRes.textContent = "Strength: 50%";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});
strengthGAN.addEventListener("click", function () {
    modelSpanRes.textContent = "Strength: GAN";
    modelDropdownRes.style.visibility = "hidden";
    hiderRestore.style.visibility = "hidden";
});

//toggle model dropdown
modelInputRes.addEventListener("click", function () {
    hiderRestore.style.visibility = "visible";
    modelDropdownRes.style.visibility = "visible";
});

// save ffmpeg params on change
ffmpegParams.addEventListener("change", function () {
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.interpolation[0].params = ffmpegParams.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Params written to project file.");
});

ffmpegParamsUpscale.addEventListener("change", function () {
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.upscaling[0].params = ffmpegParamsUpscale.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Params written to project file.");
});

ffmpegParamsRestoration.addEventListener("change", function () {
    var currentProject = sessionStorage.getItem("currentProject");
    const data = JSON.parse(fs.readFileSync(currentProject));
    data.restoration[0].params = ffmpegParamsRestoration.value;
    fs.writeFileSync(currentProject, JSON.stringify(data, null, 4));
    console.log("Params written to project file.");
});

if (localStorage.getItem('pro') == 'false') {
    console.log('Adjusting UI for lite version..')

    document.getElementById('cain-trt').style.color = '#4f4f4f';
    document.getElementById('cain-trt').style.pointerEvents = 'none';
    document.getElementById('rife-trt').style.color = '#4f4f4f';
    document.getElementById('rife-trt').style.pointerEvents = 'none';
    document.getElementById('gmfss').style.color = '#4f4f4f';
    document.getElementById('gmfss').style.pointerEvents = 'none';
    document.getElementById('gmfss-trt').style.color = '#4f4f4f';
    document.getElementById('gmfss-trt').style.pointerEvents = 'none';

    document.getElementById('shufflecugan').style.color = '#4f4f4f';
    document.getElementById('shufflecugan').style.pointerEvents = 'none';
    document.getElementById('realesrgan-tensorrt').style.color = '#4f4f4f';
    document.getElementById('realesrgan-tensorrt').style.pointerEvents = 'none';
    document.getElementById('realcugan-tensorrt').style.color = '#4f4f4f';
    document.getElementById('realcugan-tensorrt').style.pointerEvents = 'none';
    document.getElementById('swinir').style.color = '#4f4f4f';
    document.getElementById('swinir').style.pointerEvents = 'none';

    document.getElementById('dpir').style.color = '#4f4f4f';
    document.getElementById('dpir').style.pointerEvents = 'none';
    document.getElementById('anime-video').style.color = '#4f4f4f';
    document.getElementById('anime-video').style.pointerEvents = 'none';

    document.getElementById('pro-badge-user').style.visibility = 'hidden';
}