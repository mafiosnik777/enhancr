const fs = require("fs");
const sysinfo = require('systeminformation');
const path = require("path");
const shell = require("electron").shell;
const remote = require('@electron/remote');
const { BrowserWindow } = remote;
const nvencDevice = require("@mafiosnik/nvenc-codecs");

var saveBtn = document.getElementById("settings-button");
var preview = document.getElementById("preview");
let cacheInputText = document.getElementById('cache-input-text');

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

function saveSettings() {
  var previewCheck = document.getElementById("preview-check").checked;
  var blurCheck = document.getElementById("oled-mode").checked;
  var rpcCheck = document.getElementById("toggle-rpc").checked;

  var rifeTtaCheck = document.getElementById("rife-tta-check").checked;
  var rifeUhdCheck = document.getElementById("rife-uhd-check").checked;
  var fp16Check = document.getElementById("fp16-check").checked;

  var scCheck = document.getElementById("sc-check").checked;
  var skipCheck = document.getElementById("skip-check").checked;

  var numStreams = document.getElementById("num-streams").value;
  var denoiseCheck = document.getElementById('denoise-strength').value;
  var deblockCheck = document.getElementById('deblock-strength').value;

  var tileResolution = document.getElementById("tile-res").value;
  var tilingCheck = document.getElementById("tiling-check").checked;

  var shapeResolution = document.getElementById("shape-res").value;
  var shapesCheck = document.getElementById("shape-check").checked;
  var customModelCheck = document.getElementById("custom-model-check").checked;
  var pythonCheck = document.getElementById("python-check").checked;
  var trimCheck = document.getElementById("trim-check").checked;
  var hwEncodeCheck = document.getElementById("hwencode-check").checked;
  var sensitivityCheck = document.getElementById("sensitivity-check").checked;
  var sensitivity = document.getElementById("sensitivity").checked;
  var hwEncodeCheck = document.getElementById("hwencode-check").checked;
  var unsupportedCheck = document.getElementById("unsupported-check").checked;

  var theme = sessionStorage.getItem('theme');

  preview.classList.remove("animate__delay-4s");

  if (previewCheck == true) {
    preview.style.display = "block";
    sessionStorage.setItem('previewSetting', 'true ');
  } else {
    preview.style.display = "none";
    sessionStorage.setItem('previewSetting', 'false');
  }

  var settings = {
    settings: [
      {
        preview: previewCheck,
        disableBlur: blurCheck,
        rpc: rpcCheck,
        theme: theme,
        rifeTta: rifeTtaCheck,
        rifeUhd: rifeUhdCheck,
        sc: scCheck,
        skip: skipCheck,
        fp16: fp16Check,
        num_streams: numStreams,
        denoiseStrength: denoiseCheck,
        deblockStrength: deblockCheck,
        tileRes: tileResolution,
        tiling: tilingCheck,
        temp: cacheInputText.textContent,
        shapeRes: shapeResolution,
        shapes: shapesCheck,
        trimAccurate: trimCheck,
        hwEncode: hwEncodeCheck,
        sensitivityValue: sensitivity,
        sensitivity: sensitivityCheck,
        customModel: customModelCheck,
        unsupportedEngines: unsupportedCheck,
        systemPython: pythonCheck,
        language: "english"
      },
    ],
  };

  var data = JSON.stringify(settings);
  fs.writeFile(path.join(appDataPath, '/.enhancr/settings.json'), data, (err) => {
    if (err) {
      console.log("Error writing file", err);
    } else {
      console.log("JSON data is written to the file successfully");
    }
  });
  sessionStorage.setItem('settingsSaved', true);
}

saveBtn.addEventListener("click", saveSettings);

const os = require('os');
const { ipcRenderer, ipcMain } = require("electron");

function getTmpPath() {
  if (process.platform == 'win32') {
    return os.tmpdir() + "\\enhancr\\";
  } else {
    return os.tmpdir() + "/enhancr/";
  }
}
let temp = getTmpPath();

// Read settings.json on launch
fs.readFile(path.join(appDataPath, '/.enhancr/settings.json'), (err, settings) => {
  if (err) throw err;
  let json = JSON.parse(settings);
  // Set values
  document.getElementById("preview-check").checked = json.settings[0].preview;
  if (document.getElementById("preview-check").checked == true) {
    preview.style.display = "block";
    sessionStorage.setItem('previewSetting', 'true');
  } else {
    preview.style.display = "none";
    sessionStorage.setItem('previewSetting', 'false');
  }
  if (json.settings[0].disableBlur === false) {
    document.getElementById("oled-mode").checked = false;
  } else {
    document.getElementById("oled-mode").checked = true;
  }
  try {
    document.getElementById("toggle-rpc").checked = json.settings[0].rpc || true;
    document.getElementById("rife-tta-check").checked = json.settings[0].rifeTta || false;
    document.getElementById("rife-uhd-check").checked = json.settings[0].rifeUhd || false;
    document.getElementById("fp16-check").checked = json.settings[0].fp16 || true;
    document.getElementById("num-streams").value = json.settings[0].num_streams || 2;
    document.getElementById("denoise-strength").value = json.settings[0].denoiseStrength || 20;
    document.getElementById("deblock-strength").value = json.settings[0].deblockStrength || 15;
    document.getElementById("tile-res").value = json.settings[0].tileRes || '512x512';
    document.getElementById("tiling-check").checked = json.settings[0].tiling || false;
    document.getElementById("shape-res").value = json.settings[0].shapeRes || '1080x1920';
    document.getElementById("shape-check").checked = json.settings[0].shapes || false;
    document.getElementById("custom-model-check").checked = json.settings[0].customModel || false;
    document.getElementById("python-check").checked = json.settings[0].systemPython || false;
    document.getElementById("sc-check").checked = json.settings[0].sc || true;
    document.getElementById("skip-check").checked = json.settings[0].skip || false;
    document.getElementById("cache-input-text").textContent = json.settings[0].temp || temp;
    document.getElementById("trim-check").checked = json.settings[0].trimAccurate || false;
    document.getElementById("hwencode-check").checked = json.settings[0].hwEncode || false;
    document.getElementById("sensitivity").checked = json.settings[0].sensitivityValue || 0.100;
    document.getElementById("sensitivity-check").checked = json.settings[0].sensitivity || false;
    document.getElementById("unsupported-check").checked = json.settings[0].unsupportedEngines || false;
  } catch (error) {
    console.error(error);
    console.log('Incompatible settings.json detected, saving settings to overwrite incompatible one.')
    saveSettings();
  }
});

const bytesToMegaBytes = bytes => bytes / (1024 ** 2);

var cpu = document.getElementById("settings-cpu-span");
var gpu = document.getElementById("settings-gpu-span");
var mem = document.getElementById("settings-memory-span");
var osSpan = document.getElementById("settings-os-span");
var display = document.getElementById("settings-display-span");

function detectHWEncodingCaps() {
  var waitUntilGPUInitialization = setInterval(() => {
    if (sessionStorage.getItem('hasNVIDIA') != null) {
      if (sessionStorage.getItem('hasNVIDIA') == "true") {
        if (nvencDevice.supports('AV1')) {
          document.getElementById('AV1-hw').innerHTML = '<i class="fa-solid fa-check"></i> AV1';
          document.getElementById('AV1-hw').classList.add('green');
          clearInterval(waitUntilGPUInitialization);
        }
        if (nvencDevice.supports('HEVC')) {
          document.getElementById('H265-hw').innerHTML = '<i class="fa-solid fa-check"></i> H265';
          document.getElementById('H265-hw').classList.add('green');
          clearInterval(waitUntilGPUInitialization);
        }
        if (nvencDevice.supports('H264')) {
          document.getElementById('H264-hw').innerHTML = '<i class="fa-solid fa-check"></i> H264';
          document.getElementById('H264-hw').classList.add('green');
          clearInterval(waitUntilGPUInitialization);
        }
      } else {
        document.getElementById('AV1-hw').innerHTML = '<i class="fa-solid fa-question"></i> AV1';
        document.getElementById('AV1-hw').classList.add('yellow');
        document.getElementById('H265-hw').innerHTML = '<i class="fa-solid fa-question"></i> H265';
        document.getElementById('H265-hw').classList.add('yellow');
        document.getElementById('H264-hw').innerHTML = '<i class="fa-solid fa-question"></i> H264';
        document.getElementById('H264-hw').classList.add('yellow');
        clearInterval(waitUntilGPUInitialization);
      }
    }
  }, 1000);
}
setTimeout(detectHWEncodingCaps, 5000);

async function loadSettingsInfo() {
  var cpuInfo = await sysinfo.cpu().then(data => cpu.innerHTML = " " + data.manufacturer + " " + data.brand + " - " + data.physicalCores + "C/" + data.cores + "T");
  var gpuInfo = sysinfo.graphics().then(data => {
    // check for gpus and set final gpu based on hierarchy
    for (let i = 0; i < data.controllers.length; i++) {
      if (data.controllers[i].vendor.includes("Intel")) {
        gpu.innerHTML = " " + data.controllers[i].model + " - " + data.controllers[i].vram + " MB";
      }
      if (data.controllers[i].vendor.includes("AMD") || data.controllers[i].vendor.includes("Advanced Micro Devices")) {
        gpu.innerHTML = " " + data.controllers[i].model + " - " + data.controllers[i].vram + " MB";
      }
      if (data.controllers[i].vendor.includes("NVIDIA")) {
        gpu.innerHTML = " " + data.controllers[i].model + " - " + data.controllers[i].vram + " MB";
      }
    }
  });
  var memInfo = await sysinfo.mem().then(data => mem.innerHTML = " " + Math.round(bytesToMegaBytes(data.total)) + " MB");
  var osInfo = await sysinfo.osInfo().then(data => osSpan.innerHTML = " " + data.distro + " " + data.arch);
  var displayInfo = await sysinfo.graphics().then(data => display.innerHTML = " " + data.displays[0].currentResX + " x " + data.displays[0].currentResY + ", " + data.displays[0].currentRefreshRate + " Hz");
}
loadSettingsInfo();

let cacheInput = document.getElementById('cache-input');
cacheInput.addEventListener('click', () => {
  ipcRenderer.send('temp-dialog');
  sessionStorage.setItem('settingsSaved', 'false');
  // Handle event reply from main process with selected temp dir
  ipcRenderer.on("temp-dir", (event, file) => {
    cacheInputText.textContent = file;
  })
})

let customModelBtn = document.getElementById('open-custom-model-folder');

customModelBtn.addEventListener('click', () => {
  remote.shell.showItemInFolder(path.join(appDataPath, '/.enhancr/models/RealESRGAN'));
})

const pageSwitcher = document.getElementById('settings-switcher');
const settingsList = document.getElementById('settings-list');
const theming = document.getElementById('theming');

const rifeSettings = document.getElementById('rife-list');
const interpolationSettings = document.getElementById('interpolation-settings-list');
const tensorrtSettings = document.getElementById('tensorrt-list');

const dpirSettings = document.getElementById('dpir-list');
const tilingSettings = document.getElementById('tiling-list');
const shapeSettings = document.getElementById('shapes-list');
const hwEncodeSettings = document.getElementById('hwencode-list');

let toggle = false;

function changePage() {
  if (pageSwitcher.innerHTML == '<span>Page 1 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span>Page 2 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    settingsList.style.visibility = 'hidden';
    theming.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'visible';
    interpolationSettings.style.visibility = 'visible';
    tensorrtSettings.style.visibility = 'hidden';
    dpirSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span>Page 2 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span>Page 3 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    settingsList.style.visibility = 'hidden';
    theming.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'hidden';
    interpolationSettings.style.visibility = 'hidden';
    dpirSettings.style.visibility = 'hidden';
    tensorrtSettings.style.visibility = 'visible';
    tilingSettings.style.visibility = 'visible';
    shapeSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span>Page 3 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span>Page 4 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    document.getElementById('trim-list').style.visibility = 'visible';
    hwEncodeSettings.style.visibility = 'visible';
    document.getElementById('sensitivity-list').style.visibility = 'visible';
    document.getElementById('unsupported-list').style.visibility = 'visible';
    tensorrtSettings.style.visibility = 'hidden';
    tilingSettings.style.visibility = 'hidden';
    shapeSettings.style.visibility = 'hidden';
  } else if (pageSwitcher.innerHTML == '<span>Page 4 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 5 / 5</span>';
    document.getElementById('realesrgan-list').style.visibility = 'visible';
    document.getElementById('cache-list').style.visibility = 'visible';
    document.getElementById('language-list').style.visibility = 'visible';
    document.getElementById('python-list').style.visibility = 'visible';
    document.getElementById('trim-list').style.visibility = 'hidden';
    hwEncodeSettings.style.visibility = 'hidden';
    document.getElementById('sensitivity-list').style.visibility = 'hidden';
    document.getElementById('unsupported-list').style.visibility = 'hidden';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 5 / 5</span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 4 / 5</span>';
    document.getElementById('realesrgan-list').style.visibility = 'hidden';
    document.getElementById('cache-list').style.visibility = 'hidden';
    document.getElementById('language-list').style.visibility = 'hidden';
    document.getElementById('python-list').style.visibility = 'hidden';
    document.getElementById('trim-list').style.visibility = 'visible';
    hwEncodeSettings.style.visibility = 'visible';
    document.getElementById('sensitivity-list').style.visibility = 'visible';
    document.getElementById('unsupported-list').style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 4 / 5</span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 3 / 5</span>';
    document.getElementById('trim-list').style.visibility = 'hidden';
    document.getElementById('unsupported-list').style.visibility = 'hidden';
    document.getElementById('sensitivity-list').style.visibility = 'hidden';
    hwEncodeSettings.style.visibility = 'hidden';
    tensorrtSettings.style.visibility = 'visible';
    tilingSettings.style.visibility = 'visible';
    shapeSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 3 / 5</span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 2 / 5</span>';
    tensorrtSettings.style.visibility = 'hidden';
    tilingSettings.style.visibility = 'hidden';
    shapeSettings.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'visible';
    interpolationSettings.style.visibility = 'visible';
    dpirSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 2 / 5</span>') {
    pageSwitcher.innerHTML = '<span>Page 1 / 5 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    rifeSettings.style.visibility = 'hidden';
    interpolationSettings.style.visibility = 'hidden';
    dpirSettings.style.visibility = 'hidden';
    settingsList.style.visibility = 'visible';
    theming.style.visibility = 'visible';
  }
}

//language switcher
const languageSelector = document.getElementById('language-selector');
const languageDropdown = document.getElementById('language-dropdown');
const hider = document.getElementById('models-hider');

const english = document.getElementById('english');
const russian = document.getElementById('russian');
const german = document.getElementById('german');
const chinese = document.getElementById('chinese');

english.addEventListener('click', () => {
  languageSelector.innerHTML = 'English <i class="fa-solid fa-angle-down small-angle"></i>';
  hider.style.visibility = 'hidden';
  languageDropdown.style.visibility = 'hidden';
});
// russian.addEventListener('click', () => {
//   languageSelector.textContent = 'Русский <i class="fa-solid fa-angle-down small-angle"></i>';
//    hider.style.visibility = 'hidden';
//    languageDropdown.style.visibility = 'hidden';
// });
// german.addEventListener('click', () => {
//   languageSelector.textContent = 'Deutsch <i class="fa-solid fa-angle-down small-angle"></i>';
//    hider.style.visibility = 'hidden';
//    languageDropdown.style.visibility = 'hidden';
// });
// chinese.addEventListener('click', () => {
//   languageSelector.textContent = '中文 <i class="fa-solid fa-angle-down small-angle"></i>';
//    hider.style.visibility = 'hidden';
//    languageDropdown.style.visibility = 'hidden';
// });

languageSelector.addEventListener('click', () => {
  hider.style.visibility = 'visible';
  languageDropdown.style.visibility = 'visible';
});

hider.addEventListener('click', () => {
  hider.style.visibility = 'hidden';
  languageDropdown.style.visibility = 'hidden';
});


pageSwitcher.addEventListener('click', changePage);

var previewToggle = document.getElementById("preview-check");
var blurToggle = document.getElementById("oled-mode");
var rpcToggle = document.getElementById("toggle-rpc");

var rifeTtaToggle = document.getElementById("rife-tta-check");
var rifeUhdToggle = document.getElementById("rife-uhd-check");
var scCheck = document.getElementById("sc-check");
var skipCheck = document.getElementById("skip-check");
var fp16Toggle = document.getElementById("fp16-check");
var streamLine = document.getElementById("num-streams");
var denoiseToggle = document.getElementById("denoise-strength");
var deblockToggle = document.getElementById("deblock-strength");

var tileRes = document.getElementById("tile-res");
var tilingCheck = document.getElementById("tiling-check");
var shapeRes = document.getElementById("shape-res");
var shapesCheck = document.getElementById("shape-check");

previewToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
blurToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
rpcToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
rifeTtaToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
rifeUhdToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
scCheck.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
skipCheck.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
fp16Toggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
streamLine.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
deblockToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
denoiseToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
shapesCheck.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
tilingCheck.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
shapeRes.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
tileRes.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
