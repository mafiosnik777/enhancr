let interpolationTab = document.getElementById("interpolation-tab"),
  interpolationBtn = document.getElementById("interpolate-side"),
  upscaleBtn = document.getElementById("upscale-side-arrow"),
  upscaleBtn2 = document.getElementById("upscale-side"),
  restoreBtn = document.getElementById("restore-side"),
  upscalingTab = document.getElementById("upscaling-tab"),
  restorationTab = document.getElementById("restoration-tab"),
  settingsBtn = document.getElementById("settings-side"),
  settingsTab = document.getElementById("settings-tab"),
  terminalTab = document.getElementById("terminal-tab"),
  modelsBtn = document.getElementById("models-side"),
  mediaContainer = document.getElementById("media-container"),
  mediaContainerUpscale = document.getElementById("media-container-upscale"),
  mediaContainerRestore = document.getElementById("media-container-restoration");

var thumbInterpolation = document.getElementById("thumb-interpolation");
var thumbUpscaling = document.getElementById("thumb-upscaling");
var thumbRestoration = document.getElementById("thumb-restoration");

var upscaleInputText = document.getElementById("upscale-input-text"),
    restoreInputText = document.getElementById("restore-input-text"),
    nomedia = document.getElementById("nomedia");

var videoInputText = document.getElementById("input-video-text");

var previewContainer = document.getElementById('preview-container');

function openModal(modal) {
  if (modal == undefined) return
  modal.classList.add('active')
  overlay.classList.add('active')
}

function removeAnimations() {
  var styles = `
.animate__animated { 
  -webkit-animation-fill-mode: none !important;
  animation-fill-mode: none !important;
`
  var styleSheet = document.createElement("style")
  styleSheet.innerText = styles
  document.head.appendChild(styleSheet)
  interpolationTab.classList.remove('animate__animated')
}

var betaModal = document.getElementById('modal-beta');

previewContainer.addEventListener('animationend', () => {
  removeAnimations();
  openModal(betaModal);
});

let saveModal = document.getElementById("modal-save");
var engineDropdown = document.getElementById("engine-dropdown");
var hider = document.getElementById("hider");
var containerDropdown = document.getElementById('container-dropdown');
var modelsDropdown = document.getElementById('model-dropdown');

const modelsHider = document.getElementById('models-hider');
const customModelDropdown = document.getElementById('custom-model-dropdown');

const processOverlay = document.getElementById('process-overlay');

function toggleInterpolation() {
  if (sessionStorage.getItem('queueTab') == 'open') {
    thumbInterpolation.style.visibility = "hidden";
  } else {
    thumbInterpolation.style.visibility = "visible";
  }
  thumbUpscaling.style.visibility = "hidden";
  thumbRestoration.style.visibility = "hidden";
  interpolationBtn.style.color = "#d0d0d0";
  modelsBtn.style.color = "rgb(129 129 129)";
  upscaleBtn.style.color = "rgb(129 129 129)";
  upscaleBtn2.style.color = "rgb(129 129 129)";
  restoreBtn.style.color = "rgb(129 129 129)";
  settingsBtn.style.color = "rgb(129 129 129)";
  terminalTab.style.display = "block";
  upscalingTab.style.display = "none";
  restorationTab.style.display = "none";
  settingsTab.style.display = "none";
  hider.style.visibility = "hidden";
  modelsHider.style.visibility = "hidden";
  customModelDropdown.style.visibility = "hidden";
  containerDropdown.style.visibility = "hidden";
  engineDropdown.style.visibility = "hidden";
  modelsDropdown.style.visibility = "hidden";
  interpolationTab.style.display = "block";
  mediaContainer.style.visibility = "visible";
  mediaContainerUpscale.style.visibility = "hidden";
  mediaContainerRestore.style.visibility = "hidden";
  sessionStorage.setItem("currentTab", "interpolation");
  if (videoInputText.textContent === "") {
    nomedia.style.visibility = "visible";
    mediaContainer.style.visibility = "hidden";
  } else {
    nomedia.style.visibility = "hidden";
    mediaContainer.style.visibility = "visible";
  }
  if (sessionStorage.getItem('settingsSaved') == 'false') {
    openModal(saveModal);
    sessionStorage.setItem('settingsSaved', 'true');
    settingsBtn.click();
  }
  if (sessionStorage.getItem('status') == 'upscaling' || sessionStorage.getItem('status') == 'interpolating') {
    processOverlay.style.visibility = 'visible';
  }
}

function toggleUpscaling() {
  if (sessionStorage.getItem('queueTab') == 'open') {
    thumbUpscaling.style.visibility = "hidden";
  } else {
    thumbUpscaling.style.visibility = "visible";
  }
  thumbInterpolation.style.visibility = "hidden";
  thumbRestoration.style.visibility = "hidden";
  upscalingTab.classList.remove("animate__fast");
  terminalTab.classList.remove("animate__delay-3s");
  interpolationBtn.style.color = "rgb(129 129 129)";
  modelsBtn.style.color = "rgb(129 129 129)";
  upscaleBtn.style.color = "#d0d0d0";
  upscaleBtn2.style.color = "#d0d0d0";
  restoreBtn.style.color = "rgb(129 129 129)";
  settingsBtn.style.color = "rgb(129 129 129)";
  terminalTab.style.display = "block";
  interpolationTab.style.display = "none";
  restorationTab.style.display = "none";
  hider.style.visibility = "hidden";
  modelsHider.style.visibility = "hidden";
  customModelDropdown.style.visibility = "hidden";
  containerDropdown.style.visibility = "hidden";
  engineDropdown.style.visibility = "hidden";
  modelsDropdown.style.visibility = "hidden";
  settingsTab.style.display = "none";
  upscalingTab.style.display = "block";
  mediaContainer.style.visibility = "hidden";
  mediaContainerUpscale.style.visibility = "visible";
  mediaContainerRestore.style.visibility = "hidden";
  sessionStorage.setItem("currentTab", "upscaling");
  if (upscaleInputText.textContent === "") {
    nomedia.style.visibility = "visible";
    mediaContainerUpscale.style.visibility = "hidden";
  } else {
    nomedia.style.visibility = "hidden";
    mediaContainerUpscale.style.visibility = "visible";
  }
  if (sessionStorage.getItem('settingsSaved') == 'false') {
    openModal(saveModal);
    sessionStorage.setItem('settingsSaved', 'true');
    settingsBtn.click();
  }
  if (sessionStorage.getItem('status') == 'upscaling' || sessionStorage.getItem('status') == 'interpolating') {
    processOverlay.style.visibility = 'visible';
  }
}

function toggleRestoration() {
  if (sessionStorage.getItem('queueTab') == 'open') {
    thumbRestoration.style.visibility = "hidden";
  } else {
    thumbRestoration.style.visibility = "visible";
  }
  thumbInterpolation.style.visibility = "hidden";
  thumbUpscaling.style.visibility = "hidden";
  upscalingTab.classList.remove("animate__fast");
  terminalTab.classList.remove("animate__delay-3s");
  interpolationBtn.style.color = "rgb(129 129 129)";
  modelsBtn.style.color = "rgb(129 129 129)";
  upscaleBtn.style.color = "rgb(129 129 129)";
  upscaleBtn2.style.color = "rgb(129 129 129)";
  restoreBtn.style.color = "#d0d0d0";
  settingsBtn.style.color = "rgb(129 129 129)";
  terminalTab.style.display = "block";
  interpolationTab.style.display = "none";
  hider.style.visibility = "hidden";
  modelsHider.style.visibility = "hidden";
  customModelDropdown.style.visibility = "hidden";
  containerDropdown.style.visibility = "hidden";
  engineDropdown.style.visibility = "hidden";
  modelsDropdown.style.visibility = "hidden";
  settingsTab.style.display = "none";
  upscalingTab.style.display = "none";
  restorationTab.style.display = "block"
  mediaContainer.style.visibility = "hidden";
  mediaContainerUpscale.style.visibility = "hidden";
  mediaContainerRestore.style.visibility = "visible";
  sessionStorage.setItem("currentTab", "restoration");
  if (restoreInputText.textContent === "") {
    nomedia.style.visibility = "visible";
    mediaContainerRestore.style.visibility = "hidden";
  } else {
    nomedia.style.visibility = "hidden";
    mediaContainerRestore.style.visibility = "visible";
  }
  if (sessionStorage.getItem('settingsSaved') == 'false') {
    openModal(saveModal);
    sessionStorage.setItem('settingsSaved', 'true');
    settingsBtn.click();
  }
  if (sessionStorage.getItem('status') == 'upscaling' || sessionStorage.getItem('status') == 'interpolating' || sessionStorage.getItem('status') == 'restoring') {
    processOverlay.style.visibility = 'visible';
  }
}

let settingsSwitcher = document.getElementById('settings-switcher');

function toggleModels() {
  settingsTab.classList.remove("animate__fast");
  terminalTab.classList.remove("animate__delay-3s");
  interpolationBtn.style.color = "rgb(129 129 129)";
  upscaleBtn.style.color = "rgb(129 129 129)";
  upscaleBtn2.style.color = "rgb(129 129 129)";
  restoreBtn.style.color = "rgb(129 129 129)";
  settingsBtn.style.color = "#d0d0d0";
  modelsBtn.style.color = "rgb(129 129 129)";
  terminalTab.style.display = "none";
  interpolationTab.style.display = "none";
  hider.style.visibility = "hidden";
  modelsHider.style.visibility = "hidden";
  customModelDropdown.style.visibility = "hidden";
  containerDropdown.style.visibility = "hidden";
  engineDropdown.style.visibility = "hidden";
  modelsDropdown.style.visibility = "hidden";
  upscalingTab.style.display = "none";
  restorationTab.style.display = "none";
  settingsTab.style.display = "block";
  settingsSwitcher.innerHTML = '<span><i class="fa-solid fa-arrow-left" id="arrow-left"></i> Page 4 / 4</span>';
  document.getElementById('realesrgan-list').style.visibility = 'visible';
  document.getElementById('language-list').style.visibility = 'visible';
  document.getElementById('shapes-list').style.visibility = 'hidden';
  document.getElementById('tiling-list').style.visibility = 'hidden';
  document.getElementById('tensorrt-list').style.visibility = 'hidden';
  document.getElementById('rife-list').style.visibility = 'hidden';
  document.getElementById('cain-list').style.visibility = 'hidden';
  document.getElementById('dpir-list').style.visibility = 'hidden';
  document.getElementById('settings-list').style.visibility = 'hidden';
  document.getElementById('theming').style.visibility = 'hidden';
  if (sessionStorage.getItem('settingsSaved') == 'false') {
    openModal(saveModal);
    sessionStorage.setItem('settingsSaved', 'true');
    settingsBtn.click();
  }
  processOverlay.style.visibility = 'hidden';
}

function toggleSettings() {
  settingsTab.classList.remove("animate__fast");
  terminalTab.classList.remove("animate__delay-3s");
  interpolationBtn.style.color = "rgb(129 129 129)";
  modelsBtn.style.color = "rgb(129 129 129)";
  upscaleBtn.style.color = "rgb(129 129 129)";
  upscaleBtn2.style.color = "rgb(129 129 129)";
  restoreBtn.style.color = "rgb(129 129 129)";
  settingsBtn.style.color = "#d0d0d0";
  hider.style.visibility = "hidden";
  modelsHider.style.visibility = "hidden";
  customModelDropdown.style.visibility = "hidden";
  containerDropdown.style.visibility = "hidden";
  engineDropdown.style.visibility = "hidden";
  modelsDropdown.style.visibility = "hidden";
  terminalTab.style.display = "none";
  interpolationTab.style.display = "none";
  upscalingTab.style.display = "none";
  restorationTab.style.display = "none";
  settingsTab.style.display = "block";
  processOverlay.style.visibility = 'hidden';
}

interpolationBtn.addEventListener("click", toggleInterpolation);
upscaleBtn.addEventListener("click", toggleUpscaling);
restoreBtn.addEventListener("click", toggleRestoration);
settingsBtn.addEventListener("click", toggleSettings);
modelsBtn.addEventListener("click", toggleModels);
