const fs = require("fs");
const sysinfo = require('systeminformation');
const path = require("path");
const shell = require("electron").shell;
const remote = require('@electron/remote');
const { BrowserWindow } = remote;

var saveBtn = document.getElementById("settings-button");
var preview = document.getElementById("preview");

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

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
    document.getElementById("toggle-rpc").checked = json.settings[0].rpc;
    document.getElementById("rife-tta-check").checked = json.settings[0].rifeTta;
    document.getElementById("rife-uhd-check").checked = json.settings[0].rifeUhd;
    document.getElementById("rife-sc-check").checked = json.settings[0].rifeSc;
    document.getElementById("cain-sc-check").checked = json.settings[0].cainSc;
    document.getElementById("fp16-check").checked = json.settings[0].fp16;
    document.getElementById("num-streams").value = json.settings[0].num_streams;
    document.getElementById("denoise-strength").value = json.settings[0].denoiseStrength;
    document.getElementById("deblock-strength").value = json.settings[0].deblockStrength;
    document.getElementById("tile-res").value = json.settings[0].tileRes;
    document.getElementById("tiling-check").checked = json.settings[0].tiling;
    document.getElementById("shape-res").value = json.settings[0].shapeRes;
    document.getElementById("shape-check").checked = json.settings[0].shapes;
    document.getElementById("custom-model-check").checked = json.settings[0].customModel;
    document.getElementById("trim-check").checked = json.settings[0].trimAccurate;
    document.getElementById("python-check").checked = json.settings[0].systemPython;
  } catch (error) {
    console.error(error);
    console.log('Incompatible settings.json detected, save settings to overwrite.')
  }

});

const bytesToMegaBytes = bytes => bytes / (1024 ** 2);

var cpu = document.getElementById("settings-cpu-span");
var gpu = document.getElementById("settings-gpu-span");
var mem = document.getElementById("settings-memory-span");
var os = document.getElementById("settings-os-span");
var display = document.getElementById("settings-display-span");

async function loadSettingsInfo() {
  var cpuInfo = await sysinfo.cpu().then(data => cpu.innerHTML = " " + data.manufacturer + " " + data.brand + " - " + data.physicalCores + "C/" + data.cores + "T");
  var gpuInfo = await sysinfo.graphics().then(data => gpu.innerHTML = " " + data.controllers[0].model + " - " + data.controllers[0].vram + " MB");
  var memInfo = await sysinfo.mem().then(data => mem.innerHTML = " " + Math.round(bytesToMegaBytes(data.total)) + " MB");
  var osInfo = await sysinfo.osInfo().then(data => os.innerHTML = " " + data.distro + " " + data.arch);
  var displayInfo = await sysinfo.graphics().then(data => display.innerHTML = " " + data.displays[0].currentResX + " x " + data.displays[0].currentResY + ", " + data.displays[0].currentRefreshRate + " Hz");
}
loadSettingsInfo();

function saveSettings() {
  var previewCheck = document.getElementById("preview-check").checked;
  var blurCheck = document.getElementById("oled-mode").checked;
  var rpcCheck = document.getElementById("toggle-rpc").checked;

  var rifeTtaCheck = document.getElementById("rife-tta-check").checked;
  var rifeUhdCheck = document.getElementById("rife-uhd-check").checked;
  var rifeScCheck = document.getElementById("rife-sc-check").checked;
  var cainScCheck = document.getElementById("cain-sc-check").checked;
  var fp16Check = document.getElementById("fp16-check").checked;

  var numStreams = document.getElementById("num-streams").value;
  var denoiseCheck = document.getElementById('denoise-strength').value;
  var deblockCheck = document.getElementById('deblock-strength').value;

  var tileResolution = document.getElementById("tile-res").value;
  var tilingCheck = document.getElementById("tiling-check").checked;

  var shapeResolution = document.getElementById("shape-res").value;
  var shapesCheck = document.getElementById("shape-check").checked;
  var customModelCheck = document.getElementById("custom-model-check").checked;
  var trimCheck = document.getElementById("trim-check").checked;
  var pythonCheck = document.getElementById("python-check").checked;

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
        rifeSc: rifeScCheck,
        cainSc: cainScCheck,
        fp16: fp16Check,
        num_streams: numStreams,
        denoiseStrength: denoiseCheck,
        deblockStrength: deblockCheck,
        tileRes: tileResolution,
        tiling: tilingCheck,
        shapeRes: shapeResolution,
        shapes: shapesCheck,
        customModel: customModelCheck,
        trimAccurate: trimCheck,
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

let customModelBtn = document.getElementById('open-custom-model-folder');

customModelBtn.addEventListener('click', () => {
  remote.shell.showItemInFolder(path.join(appDataPath, '/.enhancr/models/RealESRGAN'));
})

const pageSwitcher = document.getElementById('settings-switcher');
const settingsList = document.getElementById('settings-list');
const theming = document.getElementById('theming');

const rifeSettings = document.getElementById('rife-list');
const cainSettings = document.getElementById('cain-list');
const tensorrtSettings = document.getElementById('tensorrt-list');

const dpirSettings = document.getElementById('dpir-list');
const tilingSettings = document.getElementById('tiling-list');
const shapeSettings = document.getElementById('shapes-list');

let toggle = false;

function changePage() {
  if (pageSwitcher.innerHTML == '<span>Page 1 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span>Page 2 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    settingsList.style.visibility = 'hidden';
    theming.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'visible';
    cainSettings.style.visibility = 'visible';
    tensorrtSettings.style.visibility = 'hidden';
    dpirSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span>Page 2 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span>Page 3 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    settingsList.style.visibility = 'hidden';
    theming.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'hidden';
    cainSettings.style.visibility = 'hidden';
    dpirSettings.style.visibility = 'hidden';
    tensorrtSettings.style.visibility = 'visible';
    tilingSettings.style.visibility = 'visible';
    shapeSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span>Page 3 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 4 / 4</span>'
    tensorrtSettings.style.visibility = 'hidden';
    tilingSettings.style.visibility = 'hidden';
    shapeSettings.style.visibility = 'hidden';
    document.getElementById('realesrgan-list').style.visibility = 'visible';
    document.getElementById('trim-list').style.visibility = 'visible';
    document.getElementById('language-list').style.visibility = 'visible';
    document.getElementById('python-list').style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 4 / 4</span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 3 / 4</span>';
    tensorrtSettings.style.visibility = 'visible';
    tilingSettings.style.visibility = 'visible';
    shapeSettings.style.visibility = 'visible';
    document.getElementById('realesrgan-list').style.visibility = 'hidden';
    document.getElementById('trim-list').style.visibility = 'hidden';
    document.getElementById('language-list').style.visibility = 'hidden';
    document.getElementById('python-list').style.visibility = 'hidden';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 3 / 4</span>') {
    pageSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 2 / 4</span>';
    tensorrtSettings.style.visibility = 'hidden';
    tilingSettings.style.visibility = 'hidden';
    shapeSettings.style.visibility = 'hidden';
    rifeSettings.style.visibility = 'visible';
    cainSettings.style.visibility = 'visible';
    dpirSettings.style.visibility = 'visible';
  } else if (pageSwitcher.innerHTML == '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 2 / 4</span>') {
    pageSwitcher.innerHTML = '<span>Page 1 / 4 <i class="fa-solid fa-arrow-right" id="arrow-right"></i></span>'
    rifeSettings.style.visibility = 'hidden';
    cainSettings.style.visibility = 'hidden';
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
var rifeScToggle = document.getElementById("rife-sc-check");
var cainScToggle = document.getElementById("cain-sc-check");
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
rifeScToggle.addEventListener('click', function () {
  sessionStorage.setItem('settingsSaved', 'false');
})
cainScToggle.addEventListener('click', function () {
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
