const { PromisePool } = require('@supercharge/promise-pool')
const path = require("path");
const remote = require('@electron/remote');
const axios = require('axios');
const fs = require('fs-extra');
const os = require('os');

const { spawn } = require('child_process');
const { clipboard } = require('electron')
const { ipcRenderer, webFrame } = require("electron");

const enhancr = require('./js/helper.js')
const Interpolation = require('./js/interpolation.js')
const Upscaling = require('./js/upscaling.js')
const Restoration = require('./js/restoration.js')
const Preview = require('./js/preview.js')
const Process = require('./js/process.js')
const Media = require('./js/media.js')
const ThemeEngine = require('./js/themeengine.js')

const { BrowserWindow } = remote;
let win = BrowserWindow.getFocusedWindow();

var terminal = document.getElementById("terminal-text");

const interpolationBtn = document.getElementById('interpolate-button-text');
const upscalingBtn = document.getElementById('upscaling-button-text');
const restoreBtn = document.getElementById('restore-button-text');
const queueBadge = document.getElementById('queue-badge');

const find = require('find-process');
const shell = require("electron").shell;
const ffmpeg = require('fluent-ffmpeg');

const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const ffprobePath = require('@ffprobe-installer/ffprobe').path;
ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);

let queueTab = document.getElementById('queue');
let queueBtn = document.getElementById('top-bar-queue');

var progressSpan = document.getElementById('progress-span');

var terminal = document.getElementById("terminal-text");

const exportBtn = document.getElementById('export-btn');

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

function getTmpPath() {
  if (process.platform == 'win32') {
    return os.tmpdir() + "\\enhancr\\";
  } else {
    return os.tmpdir() + "/enhancr/";
  }
}

var tempPath = getTmpPath();

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

let queue = [];

function renderQueueItem() {
  queueTab.innerHTML = '';

  if (queue.length != 0 && queueTab.style.visibility == 'visible') document.getElementById('clear-queue-btn').style.visibility = 'visible';

  function setScreenshot(i) {
    setTimeout(function () {
      webFrame.clearCache()
      let thumb = document.getElementById(`queue-thumb${i}`);
      thumb.setAttribute('src', path.join(appDataPath, `/.enhancr/thumbs/queue${i}.png`));
    }, 1000)
  }

  queue.forEach(function (queueItem, i = 0) {
    // create queue item container
    let container = document.createElement('div');
    container.classList.add('queue-item')
    container.setAttribute('id', `queue${i}`);
    queueTab.append(container);
    // get thumbnail
    ffmpeg(queueItem.file).screenshots({
      count: 1,
      filename: `queue${i}.png`,
      folder: path.join(appDataPath, '/.enhancr/thumbs'),
      size: '426x240'
    });
    webFrame.clearCache()
    let thumb = document.createElement('img');
    thumb.classList.add('queue-thumb');
    thumb.setAttribute('id', `queue-thumb${i}`);
    thumb.setAttribute('src', './assets/queue-load.gif')
    setScreenshot(i);
    container.append(thumb);
    // create info container
    let info = document.createElement('div');
    info.classList.add('queue-item-info')
    info.setAttribute('id', `queue-info${i}`);
    container.append(info);
    // create queue item title
    let item = document.createElement('div');
    item.classList.add('queue-item-title');
    item.setAttribute('id', `queue-title${i}`);
    info.append(item);
    item.innerHTML = path.basename(queueItem.file);
    // create queue item path
    let queuePath = document.createElement('div');
    queuePath.classList.add('queue-item-path');
    queuePath.setAttribute('id', `queue-path${i}`)
    info.append(queuePath);
    queuePath.innerHTML = queueItem.file;
    // create queue item mode
    let mode = document.createElement('div');
    mode.classList.add('queue-item-mode');
    mode.setAttribute('id', `queue-mode${i}`);
    info.append(mode);
    if (queueItem.mode == 'interpolation') {
      mode.innerHTML = '<i class="fa-solid fa-clone"></i> Interpolation';
    } else if (queueItem.mode == 'upscaling') {
      mode.innerHTML = '<i class="fa-solid fa-up-right-and-down-left-from-center"></i> Upscaling';
    } else {
      mode.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Restoration';
    }
    // create queue progress bar
    let progress = document.createElement('div');
    progress.classList.add('queue-item-progress');
    progress.setAttribute('id', `queue-progress${i}`);
    info.append(progress);
    // create percentage
    let percent = document.createElement('div');
    percent.classList.add('queue-item-percent');
    percent.setAttribute('id', `queue-percent${i}`);
    info.append(percent);
    percent.innerHTML = '0%';
    if (queueItem.status == '1') {
      percent.innerHTML = '100%';
    }
    // create remove button
    let remove = document.createElement('i');
    remove.classList.add('fa-solid');
    if (queueItem.status == '1') {
      remove.classList.add('fa-check');
      remove.style.color = '#50C878';
      remove.classList.add('queue-item-done');
    } else {
      remove.classList.add('fa-trash-can');
      remove.classList.add('queue-item-remove');
    }
    remove.setAttribute('id', `queue-item-remove${i}`);
    info.append(remove);
    // create context menu button
    let dots = document.createElement('i');
    dots.classList.add('fa-solid');
    dots.classList.add('fa-ellipsis');
    dots.classList.add('queue-item-dots');
    dots.setAttribute('id', `queue-item-dots${i}`)
    info.append(dots);
    // create context menu
    let menu = document.createElement('ul');
    menu.classList.add('context-menu-queue');
    info.append(menu);
    let listItem1 = document.createElement('li');
    let listItem2 = document.createElement('li');
    let listItem3 = document.createElement('li');
    menu.setAttribute('id', `context-menu-queue${i}`);
    if (queue[i].mode == 'restoration') {
      menu.style.marginLeft = '2%';
    }
    menu.append(listItem1);
    menu.append(listItem2);
    menu.append(listItem3);
    let play = document.createElement('button');
    play.classList.add('menu__item');
    play.classList.add('context-item0');
    play.setAttribute('id', `context-item0-queue${i}`);
    listItem1.append(play);
    play.innerHTML = "<i class='fa-solid fa-link'></i> Chain"
    let openPath = document.createElement('button');
    openPath.classList.add('menu__item');
    openPath.classList.add('context-item1');
    openPath.setAttribute('id', `context-item1-queue${i}`);
    listItem2.append(openPath);
    openPath.innerHTML = "<i class='fa-solid fa-pencil'></i> Output name"
    let share = document.createElement('button');
    share.classList.add('menu__item');
    share.classList.add('context-item2');
    share.setAttribute('id', `context-item2-queue${i}`);
    listItem3.append(share);
    share.innerHTML = "<i class='fa-solid fa-scissors'></i> Trim"
    // create progress overlay
    let progressOverlay = document.createElement('div');
    progressOverlay.classList.add('queue-item-progress-overlay');
    progressOverlay.setAttribute('id', `queue-item-progress-overlay${i}`);
    progressOverlay.style.background = ThemeEngine.getTheme(sessionStorage.getItem('theme'));
    if (queueItem.status == '1') {
      progressOverlay.style.width = '100%';
    }
    progress.append(progressOverlay);

    // dots listener
    let dotsBtn = [].slice.call(document.getElementsByClassName('queue-item-dots'));

    dotsBtn.forEach(function (button) {
      if (!(button.classList.contains('listener'))) {
        button.classList.add('listener');
        button.addEventListener('click', function () {
          let id = button.id;
          let index = id.charAt(id.length - 1);
          let thumb = document.getElementById(`queue-thumb${index}`);
          if (thumb.style.visibility == 'visible' || thumb.style.visibility == '') {
            thumb.style.visibility = 'hidden';
            let menu = document.getElementById(`context-menu-queue${index}`);
            menu.style.visibility = 'visible';
          } else {
            thumb.style.visibility = 'visible';
            let menu = document.getElementById(`context-menu-queue${index}`);
            menu.style.visibility = 'hidden';
          }
        })
      }
    });

    // context menu listeners

    //chaining / play video
    let context0 = [].slice.call(document.getElementsByClassName('context-item0'));

    context0.forEach(function (item) {
      if (!(item.classList.contains('listener'))) {
        item.classList.add('listener')
        item.addEventListener('click', function () {
          let id = item.id;
          let index = id.charAt(id.length - 1);
          let chainModal = document.getElementById('modal-chain');
          let mode = queue[index].mode;
          if (item.innerHTML == '<i class="fa-solid fa-film"></i> Play video') {
            let model;
            let outputPath;
            if (!(sessionStorage.getItem(`out${index}`) == null)) {
              outputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
              shell.openPath(outputPath);
            } else if (queue[index].mode == 'interpolation') {
              if (queue[index].engine == "Channel Attention - CAIN (NCNN)") {
                model = "CAIN"
              } else if (queue[index].engine == "Optical Flow - RIFE (NCNN)") {
                model = "RIFE"
              } else if (queue[index].engine == "Optical Flow - RIFE (TensorRT)") {
                model = "RIFE"
              } else if (queue[index].engine == "Channel Attention - CAIN (TensorRT)") {
                model = "CAIN"
              } else if (queue[index].engine == "GMFlow - GMFSS (PyTorch") {
                model = "GMFSS"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
              shell.openPath(outputPath);
            } else if (queue[index].mode == 'upscaling') {
              if (queue[index].engine == "Upscaling - RealESRGAN (TensorRT)" || queue[index].engine == "Upscaling - RealESRGAN (NCNN)") {
                model = "RealESRGAN"
              } else if (queue[index].engine == "Upscaling - waifu2x (TensorRT)") {
                model = "waifu2x"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
              shell.openPath(outputPath);
            } else if (queue[index].mode == 'restoration') {
              if (queue[index].engine == "Restoration - DPIR (TensorRT)") {
                model = "DPIR"
              } else if (queue[index].engine == "Restoration - AnimeVideo (TensorRT)") {
                model = "AnimeVideo"
              } else {
                model = "AnimeVideo"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
              shell.openPath(outputPath);
            }
          } else {
            console.log(item.innerHTML)
            let chainBtnInterpolation = document.getElementById('chain-interpolation');
            let chainBtnUpscaling = document.getElementById('chain-upscaling');
            let chainBtnRestoration = document.getElementById('chain-restoration');
            if (mode == 'interpolation') {
              chainBtnInterpolation.style.display = 'none';
              chainBtnUpscaling.style.display = 'block';
              chainBtnRestoration.style.display = 'block';
            } else if (mode == 'upscaling') {
              chainBtnInterpolation.style.display = 'block';
              chainBtnUpscaling.style.display = 'none';
              chainBtnRestoration.style.display = 'block';
            } else {
              chainBtnInterpolation.style.display = 'block';
              chainBtnUpscaling.style.display = 'block';
              chainBtnRestoration.style.display = 'none';
            }
            openModal(chainModal);
            if (!(chainBtnUpscaling.classList.contains('listener'))) {
              chainBtnUpscaling.classList.add('listener');
              chainBtnUpscaling.addEventListener('click', function () {
                closeModal(chainModal);
                let upscaleTabBtn = document.getElementById('upscale-side-arrow');
                let videoInputPath;
                let model;
                var upscaleVideoInputText = document.getElementById("upscale-input-text");
                if (sessionStorage.getItem(`out${index}`) == null) {
                  if (queue[index].mode == 'interpolation') {
                    if (queue[index].engine == "Channel Attention - CAIN (NCNN)") {
                      model = "CAIN"
                    } else if (queue[index].engine == "Optical Flow - RIFE (NCNN)") {
                      model = "RIFE"
                    } else if (queue[index].engine == "Optical Flow - RIFE (TensorRT)") {
                      model = "RIFE"
                    } else if (queue[index].engine == "Channel Attention - CAIN (TensorRT)") {
                      model = "CAIN"
                    } else if (queue[index].engine == "GMFlow - GMFSS (PyTorch") {
                      model = "GMFSS"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
                  } else {
                    if (queue[index].engine == "Restoration - DPIR (TensorRT)") {
                      model = "DPIR"
                    } else if (queue[index].engine == "Restoration - AnimeVideo (TensorRT)") {
                      model = "AnimeVideo"
                    } else {
                      model = "AnimeVideo"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
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
                upscaleTabBtn.click();
              })
            }
            if (!(chainBtnRestoration.classList.contains('listener'))) {
              chainBtnRestoration.classList.add('listener');
              chainBtnRestoration.addEventListener('click', function () {
                closeModal(chainModal);
                let restoreTabBtn = document.getElementById('restore-side');
                let videoInputPath;
                let model;
                var restoreVideoInputText = document.getElementById("restore-input-text");
                if (sessionStorage.getItem(`out${index}`) == null) {
                  if (queue[index].mode == 'interpolation') {
                    if (queue[index].engine == "Channel Attention - CAIN (NCNN)") {
                      model = "CAIN"
                    } else if (queue[index].engine == "Optical Flow - RIFE (NCNN)") {
                      model = "RIFE"
                    } else if (queue[index].engine == "Optical Flow - RIFE (TensorRT)") {
                      model = "RIFE"
                    } else if (queue[index].engine == "Channel Attention - CAIN (TensorRT)") {
                      model = "CAIN"
                    } else if (queue[index].engine == "GMFlow - GMFSS (PyTorch") {
                      model = "GMFSS"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
                  } else {
                    if (queue[index].engine == "Upscaling - RealESRGAN (TensorRT)" || queue[index].engine == "Upscaling - RealESRGAN (NCNN)") {
                      model = "RealESRGAN"
                    } else if (queue[index].engine == "Upscaling - waifu2x (TensorRT)") {
                      model = "waifu2x"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
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
                restoreTabBtn.click();
              })
            }
            if (!(chainBtnInterpolation.classList.contains('listener'))) {
              chainBtnInterpolation.classList.add('listener');
              chainBtnInterpolation.addEventListener('click', function () {
                closeModal(chainModal);
                let interpolationTabBtn = document.getElementById('interpolate-side');
                let videoInputPath;
                let model;
                var interpolationVideoInputText = document.getElementById("input-video-text");
                if (sessionStorage.getItem(`out${index}`) == null) {
                  if (queue[index].mode == 'upscaling') {
                    if (queue[index].engine == "Upscaling - RealESRGAN (TensorRT)" || queue[index].engine == "Upscaling - RealESRGAN (NCNN)") {
                      model = "RealESRGAN"
                    } else if (queue[index].engine == "Upscaling - waifu2x (TensorRT)") {
                      model = "waifu2x"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
                  } else {
                    if (queue[index].engine == "Restoration - DPIR (TensorRT)") {
                      model = "DPIR"
                    } else if (queue[index].engine == "Restoration - AnimeVideo (TensorRT)") {
                      model = "AnimeVideo"
                    } else {
                      model = "AnimeVideo"
                    }
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
                sessionStorage.setItem("inputPath", videoInputPath);
                if (
                  videoInputPath.length >= 55 &&
                  path.basename(videoInputPath).length >= 55
                ) {
                  interpolationVideoInputText.textContent =
                    "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
                } else if (videoInputPath.length >= 55) {
                  interpolationVideoInputText.textContent = "../" + path.basename(videoInputPath);
                } else {
                  interpolationVideoInputText.textContent = videoInputPath;
                }
                interpolationTabBtn.click();
              })
            }
          }
        });
      }
    });

    // edit export name / show in folder
    let context1 = [].slice.call(document.getElementsByClassName('context-item1'));

    context1.forEach(function (item) {
      if (!(item.classList.contains('listener'))) {
        item.classList.add('listener');
        item.addEventListener('click', function () {
          let id = item.id;
          let index = id.charAt(id.length - 1);
          if (item.innerHTML == '<i class="fa-solid fa-folder-closed"></i> Open in folder') {
            let model;
            let outputPath;
            if (!(sessionStorage.getItem(`out${index}`) == null)) {
              outputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
              remote.shell.showItemInFolder(outputPath);
            } else if (queue[index].mode == 'interpolation') {
              if (queue[index].engine == "Channel Attention - CAIN (NCNN)") {
                model = "CAIN"
              } else if (queue[index].engine == "Optical Flow - RIFE (NCNN)") {
                model = "RIFE"
              } else if (queue[index].engine == "Optical Flow - RIFE (TensorRT)") {
                model = "RIFE"
              } else if (queue[index].engine == "Channel Attention - CAIN (TensorRT)") {
                model = "CAIN"
              } else if (queue[index].engine == "GMFlow - GMFSS (PyTorch") {
                model = "GMFSS"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
              remote.shell.showItemInFolder(outputPath);
            } else if (queue[index].mode == 'upscaling') {
              if (queue[index].engine == "Upscaling - RealESRGAN (TensorRT)" || queue[index].engine == "Upscaling - RealESRGAN (NCNN)") {
                model = "RealESRGAN"
              } else if (queue[index].engine == "Upscaling - waifu2x (TensorRT)") {
                model = "waifu2x"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
              remote.shell.showItemInFolder(outputPath);
            } else if (queue[index].mode == 'restoration') {
              if (queue[index].engine == "Restoration - DPIR (TensorRT)") {
                model = "DPIR"
              } else if (queue[index].engine == "Restoration - AnimeVideo (TensorRT)") {
                model = "AnimeVideo"
              } else {
                model = "AnimeVideo"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
              remote.shell.showItemInFolder(outputPath);
            }
          } else {
            let editModal = document.getElementById('modal-edit');
            openModal(editModal);
            let editField = document.getElementById('edit-export-name');
            editField.value = '';
            let fileName = document.getElementById(`queue-title${index}`);
            editField.placeholder = path.parse(fileName.innerHTML).name;
            let saveBtn = document.getElementById('save-edit-btn');
            saveBtn.addEventListener('click', function () {
              if (!(editField.value == '')) {
                sessionStorage.setItem(`out${index}`, editField.value);
                closeModal(editModal);
                enhancr.terminal(`Set output file name for position '${index}' in queue to: '${editField.value}'`);
              }
            })
          }
        })
      };
    });

    // open folder / share
    let context2 = [].slice.call(document.getElementsByClassName('context-item2'));

    context2.forEach(function (item) {
      if (!(item.classList.contains('listener'))) {
        item.classList.add('listener');
        item.addEventListener('click', async function () {
          let id = item.id;
          let index = id.charAt(id.length - 1);
          if (item.innerHTML == '<i class="fa-solid fa-share-nodes"></i> Share') {
            const response = await axios.get('https://api.gofile.io/getServer');
            var bestServer = response.data.data.server;
            let outputPath;
            let model;
            if (!(sessionStorage.getItem(`out${index}`) == null)) {
              outputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
            } else if (queue[index].mode == 'interpolation') {
              if (queue[index].engine == "Channel Attention - CAIN (NCNN)") {
                model = "CAIN"
              } else if (queue[index].engine == "Optical Flow - RIFE (NCNN)") {
                model = "RIFE"
              } else if (queue[index].engine == "Optical Flow - RIFE (TensorRT)") {
                model = "RIFE"
              } else if (queue[index].engine == "Channel Attention - CAIN (TensorRT)") {
                model = "CAIN"
              } else if (queue[index].engine == "GMFlow - GMFSS (PyTorch") {
                model = "GMFSS"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
            } else if (queue[index].mode == 'upscaling') {
              if (queue[index].engine == "Upscaling - RealESRGAN (TensorRT)" || queue[index].engine == "Upscaling - RealESRGAN (NCNN)") {
                model = "RealESRGAN"
              } else if (queue[index].engine == "Upscaling - waifu2x (TensorRT)") {
                model = "waifu2x"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
            } else if (queue[index].mode == 'restoration') {
              if (queue[index].engine == "Restoration - DPIR (TensorRT)") {
                model = "DPIR"
              } else if (queue[index].engine == "Restoration - AnimeVideo (TensorRT)") {
                model = "AnimeVideo"
              } else {
                model = "AnimeVideo"
              }
              outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
            }

            var fpsMeter = document.getElementById('fps-meter');
            fpsMeter.style.visibility = "hidden";
            var eta = document.getElementById('eta');
            eta.style.visibility = "hidden";

            const progressDone = document.getElementById("progress-done");

            sessionStorage.setItem('uploadStatus', 'uploading');
            let cmd = `curl -F file=@"${outputPath}" https://${bestServer}.gofile.io/uploadFile -o ${path.join(tempPath, '/upload.json')}`;
            let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'] });
            process.stdout.write('');
            term.stdout.on('data', (data) => {
              sessionStorage.setItem("uploadProgress", data);
            });
            term.stderr.on('data', (data) => {
              sessionStorage.setItem("uploadProgress", data);
            });
            term.on("close", () => {
              win.setProgressBar(-1, { mode: "none" });
              win.flashFrame(true);
              sessionStorage.setItem('uploadStatus', 'done');
              const json = JSON.parse(fs.readFileSync(path.join(tempPath, '/upload.json')));
              terminal.textContent += "\r\n[gofile.io] Upload completed: " + json.data.downloadPage;
              terminal.scrollTop = terminal.scrollHeight;
              progressSpan.textContent = `Upload complete: ${path.basename(outputPath)} | 100%`;
              var notification = new Notification("Upload completed", { icon: "./assets/enhancr.png", body: json.data.downloadPage });
              progressDone.style.width = `100%`;
              clipboard.writeText(json.data.downloadPage);
              terminal.textContent += "\r\n[gofile.io] Copied link to clipboard.";
              window.open(json.data.downloadPage, "_blank");
            });

            var progressInterval = setInterval(async function () {
              var progress = sessionStorage.getItem('uploadProgress');
              if (progress.length >= 22) {
                var trimmed = progress.trim();
                var percentage = trimmed.split(' ')[0];
                terminal.textContent += `\r\n[gofile.io] Uploading... (${percentage}%)`;
                terminal.scrollTop = terminal.scrollHeight;
                progressSpan.textContent = `Uploading ${path.basename(outputPath)} | ${percentage}%`;
                progressDone.style.width = `${percentage}%`;
                if (percentage == 100) {
                  sessionStorage.setItem('uploadStatus', 'done');
                }
                win.setProgressBar(parseInt(percentage) / 100);
                ipcRenderer.send('rpc-uploading', parseInt(percentage), path.basename(outputPath));
                if (sessionStorage.getItem('uploadStatus') == 'done') {
                  win.setProgressBar(-1);
                  win.flashFrame(true);
                  clearInterval(progressInterval);
                }
              } else {
                // do nothing
              }
            }, 1000)
          } else {
            let trimModal = document.getElementById('modal-trim');
            openModal(trimModal);
            let saveTimestamps = document.getElementById('save-timestamps-btn');
            saveTimestamps.addEventListener('click', () => {
              sessionStorage.setItem(`trim${index}`, document.getElementById('timestamp-trim').value);
              closeModal(trimModal);
            })
          }
        });
      }
    });

    // remove queue item listener
    let removeBtn = [].slice.call(document.getElementsByClassName('queue-item-remove'));

    removeBtn.forEach(function (button) {
      if (!(button.classList.contains('listener'))) {
        button.classList.add('listener');
        button.addEventListener('click', function () {
          if (!(button.classList.contains('queue-item-spin'))) {
            if (!(button.classList.contains('queue-item-done'))) {
              console.log(button.classList.contains('queue-item-spin'));
              let id = button.id;
              let index = id.charAt(id.length - 1);
              console.log(index);
              queue.splice(index, 1);
              let count = queue.reduce((a, c) => c.status == '0' ? ++a : a, 0);
              queueBadge.innerHTML = count;
              if (queue.length == 0) {
                queueTab.innerHTML = '';
                document.getElementById('clear-queue-btn').style.visibility = 'hidden';
                let blank = document.createElement('span');
                blank.setAttribute('id', `queue-blank`);
                blank.innerHTML = 'No items scheduled for processing, add items to the queue to get started.'
                queueTab.append(blank);
              } else {
                renderQueueItem();
              }
            }
          }
        });
      }
    });
    i++;
  });
}

async function addToQueueInterpolation() {
  const modelSpan = document.getElementById("model-span");
  const ffmpegParams = document.getElementById("ffmpeg-params");
  const outputContainer = document.getElementById("container-span");
  const interpolationEngine = document.getElementById("interpolation-engine-text");
  const dimensions = document.getElementById('dimensions');

  if (sessionStorage.getItem("multiInput") == "true") {
    let file = JSON.parse(sessionStorage.getItem('interpolationMultiInput'))
    for (var i=0; i< file.length; i++) {
      queue.push({
        mode: 'interpolation',
        file: file[i],
        model: modelSpan.innerHTML,
        output: sessionStorage.getItem("outputPath"),
        params: ffmpegParams.value,
        extension: outputContainer.innerHTML,
        engine: interpolationEngine.innerHTML,
        dimensions: await Media.fetchDimensions(file[i]),
        status: '0'
      })
      enhancr.terminal(`'${path.basename(file[i])}': Successfully added to Queue`);
    }
  } else {
    queue.push({
      mode: 'interpolation',
      file: sessionStorage.getItem("inputPath"),
      model: modelSpan.innerHTML,
      output: sessionStorage.getItem("outputPath"),
      params: ffmpegParams.value,
      extension: outputContainer.innerHTML,
      engine: interpolationEngine.innerHTML,
      dimensions: dimensions.innerHTML,
      status: '0'
    })
    enhancr.terminal(`'${path.basename(sessionStorage.getItem("inputPath"))}': Successfully added to Queue`);
  }

  console.log(queue);
  let count = queue.reduce((a, c) => c.status == '0' ? ++a : a, 0);
  queueBadge.innerHTML = count;
  terminal.scrollTop = terminal.scrollHeight;
  renderQueueItem()
  sessionStorage.setItem('queueLength', queue.length);
}
interpolationBtn.addEventListener('click', addToQueueInterpolation);

async function addToQueueUpscaling() {
  const scale = document.getElementById("factor-span");
  const ffmpegParams = document.getElementById("ffmpeg-params-upscale");
  const outputContainer = document.getElementById("container-span-up");
  const upscalingEngine = document.getElementById("upscale-engine-text");
  const dimensions = document.getElementById('dimensionsUp');

  if (sessionStorage.getItem("multiInput") == "true") {
    let file = JSON.parse(sessionStorage.getItem('upscalingMultiInput'))
    for (var i=0; i< file.length; i++) {
      queue.push({
        mode: 'upscaling',
        file: file[i],
        scale: (scale.innerHTML).slice(0, 1),
        output: sessionStorage.getItem("upscaleOutputPath"),
        params: ffmpegParams.value,
        extension: outputContainer.innerHTML,
        engine: upscalingEngine.innerHTML,
        dimensions: await Media.fetchDimensions(file[i]),
        status: '0'
      })
      enhancr.terminal(`'${path.basename(file[i])}': Successfully added to Queue`);
    }
  } else {
    queue.push({
      mode: 'upscaling',
      file: sessionStorage.getItem("upscaleInputPath"),
      scale: (scale.innerHTML).slice(0, 1),
      output: sessionStorage.getItem("upscaleOutputPath"),
      params: ffmpegParams.value,
      extension: outputContainer.innerHTML,
      engine: upscalingEngine.innerHTML,
      dimensions: dimensions.innerHTML,
      status: '0'
    })
    enhancr.terminal(`'${path.basename(sessionStorage.getItem("upscaleInputPath"))}': Successfully added to Queue`);
  }

  console.log(queue);
  let count = queue.reduce((a, c) => c.status == '0' ? ++a : a, 0);
  queueBadge.innerHTML = count;
  terminal.scrollTop = terminal.scrollHeight;
  renderQueueItem()
  sessionStorage.setItem('queueLength', queue.length);
}
upscalingBtn.addEventListener('click', addToQueueUpscaling);

function addToQueueRestoration() {
  const ffmpegParams = document.getElementById("ffmpeg-params-restoration");
  const outputContainer = document.getElementById("container-span-res");
  const restoreEngine = document.getElementById("restoration-engine-text");
  const modelSpan = document.getElementById("model-span-restoration");

  if (sessionStorage.getItem("multiInput") == "true") {
    let file = JSON.parse(sessionStorage.getItem('restorationMultiInput'))
    for (var i=0; i< file.length; i++) {
      queue.push({
        mode: 'restoration',
        file: file[i],
        model: modelSpan.innerHTML,
        output: sessionStorage.getItem("outputPathRestoration"),
        params: ffmpegParams.value,
        extension: outputContainer.innerHTML,
        engine: restoreEngine.innerHTML,
        status: '0'
      })
      enhancr.terminal(`'${path.basename(file[i])}': Successfully added to Queue`);
    }
  } else {
    queue.push({
      mode: 'restoration',
      file: sessionStorage.getItem("inputPathRestore"),
      model: modelSpan.innerHTML,
      output: sessionStorage.getItem("outputPathRestoration"),
      params: ffmpegParams.value,
      extension: outputContainer.innerHTML,
      engine: restoreEngine.innerHTML,
      status: '0'
    })
    enhancr.terminal(`'${path.basename(sessionStorage.getItem("inputPathRestore"))}': Successfully added to Queue`);
  }

  console.log(queue);
  let count = queue.reduce((a, c) => c.status == '0' ? ++a : a, 0);
  queueBadge.innerHTML = count;
  terminal.scrollTop = terminal.scrollHeight;
  renderQueueItem()
  sessionStorage.setItem('queueLength', queue.length);
}
restoreBtn.addEventListener('click', addToQueueRestoration);

let running = false;
try {
  async function processQueue() {
    if (running == false) {
      running = true;
      await PromisePool
        .withConcurrency(1)
        .for(queue)
        .onTaskStarted((item) => {
          if (item.status == '0') {
            if (sessionStorage.getItem('stopped') == 'false') {
              console.log('Starting queue');
              sessionStorage.setItem('currentFile', item.file);
              sessionStorage.setItem('queueIndex', queue.indexOf(item));
              exportBtn.innerHTML = '<span id="export-button"><i class="fa-solid fa-circle-stop" id="enhance-icon"></i></i> Stop processing video(s) <span id="key-shortcut">Ctrl + Enter</span></span>'
              // listen for preview
              Preview.listenForPreview();
              // initialize ui for process
              switch (item.mode) {
                case 'interpolation':
                  Process.startProcessInterpolation();
                  Media.fetchMetadata();
                  break;
                case 'upscaling':
                  sessionStorage.setItem('currentFactor', item.scale);
                  Process.startProcessUpscaling();
                  Media.fetchMetadataUpscale();
                  break;
                case 'restoration':
                  Process.startProcessRestoration();
                  Media.fetchMetadataRestore();
              }
              let queueIcon = document.getElementById(`queue-item-remove${queue.indexOf(item)}`);
              queueIcon.removeAttribute("class");
              queueIcon.classList.add('fa-solid');
              queueIcon.classList.add('fa-rotate');
              queueIcon.classList.add('fa-spin');
              queueIcon.classList.add('queue-item-spin');
            };
          }
        })
        .onTaskFinished((item) => {
          if (item.status == '0') {
            queueBadge.innerHTML = parseInt(queueBadge.innerHTML) - 1;
          }
          item.status = '1';
          let contextItem0 = document.getElementById(`context-item0-queue${queue.indexOf(item)}`);
          contextItem0.innerHTML = "<i class='fa-solid fa-film'></i> Play video";
          let contextItem1 = document.getElementById(`context-item1-queue${queue.indexOf(item)}`);
          contextItem1.innerHTML = "<i class='fa-solid fa-folder-closed'></i> Open in folder";
          let contextItem2 = document.getElementById(`context-item2-queue${queue.indexOf(item)}`);
          contextItem2.innerHTML = "<i class='fa-solid fa-share-nodes'></i> Share";
        })
        .process(async (item, index, pool) => {
          if (item.status == '0') {
            console.log(item);
            switch (item.mode) {
              case 'interpolation':
                await Interpolation.process(item.file, item.model, item.output, item.params, item.extension, item.engine, item.dimensions, sessionStorage.getItem(`out${index}`), index);
                break;
              case 'upscaling':
                await Upscaling.process(item.scale, item.dimensions, item.file, item.output, item.params, item.extension, item.engine, sessionStorage.getItem(`out${index}`), index);
                break;
              case 'restoration':
                await Restoration.process(item.file, item.model, item.output, item.params, item.extension, item.engine, sessionStorage.getItem(`out${index}`), index);
            }
          }
        })
        .then(function () {
          exportBtn.innerHTML = '<i class="fa-solid fa-film" id="enhance-icon"></i> Enhance video(s) <span id="key-shortcut">Ctrl + Enter</span>';
          if (queue.length == 0) {
            enhancr.terminal('Queue is empty. Add media to queue to get started.\r\n');
          } else {
            enhancr.terminal('Completed processing queue successfully.\r\n');
          }
          terminal.scrollTop = terminal.scrollHeight;
          running = false;
          if (sessionStorage.getItem('stopped') == 'true') {
            sessionStorage.setItem('stopped', 'false');
            // set stop flag for process.js
            sessionStorage.setItem('stopFlag', 'true');
          }
        })
        
    } else {
      sessionStorage.setItem('stopped', 'true');
      find('name', 'VSPipe', false).then(function (list) {
        var i;
        for (i = 0; i < list.length; i++) {
          process.kill(list[i].pid);
        }
      }).then(function () {
        exportBtn.innerHTML = '<i class="fa-solid fa-film" id="enhance-icon"></i> Enhance video(s) <span id="key-shortcut">Ctrl + Enter</span>';
        enhancr.terminal('Stopped queue successfully.\r\n')
        ipcRenderer.send('rpc-done');
        terminal.scrollTop = terminal.scrollHeight;
        running = false;
      });
    }
  };
  exportBtn.addEventListener('click', processQueue);
} catch (error) {
  console.error(error)
}

// render queue tab
queueTab.style.visibility = 'hidden';
async function toggleQueueTab() {
  if (queueTab.style.visibility == 'hidden') {
    sessionStorage.setItem('queueTab', 'open');
    queueTab.style.visibility = 'visible';
    if (queue.length != 0) document.getElementById('clear-queue-btn').style.visibility = 'visible';
    queueBtn.style.background = 'rgba(190, 190, 190, 0.378)';
    let thumbs = [].slice.call(document.getElementsByClassName('queue-thumb'));
    thumbs.forEach(function (thumb) {
      thumb.style.visibility = 'visible';
    });
    document.getElementById('preview-text').style.visibility = 'hidden';
    document.getElementById('thumb-interpolation').style.visibility = 'hidden';
    document.getElementById('thumb-upscaling').style.visibility = 'hidden';
    document.getElementById('thumb-restoration').style.visibility = 'hidden';
    document.getElementById('timecode').style.visibility = 'hidden';
    document.getElementById('framecount').style.visibility = 'hidden';
    document.getElementById('preview').style.visibility = 'hidden';
  } else {
    sessionStorage.setItem('queueTab', 'closed');
    queueTab.style.visibility = 'hidden';
    document.getElementById('clear-queue-btn').style.visibility = 'hidden';
    queueBtn.style.background = 'rgba(190, 190, 190, 0.1)';
    let menu = [].slice.call(document.getElementsByClassName('context-menu-queue'));
    menu.forEach(function (contextMenu) {
      contextMenu.style.visibility = 'hidden';
    });
    let thumbs = [].slice.call(document.getElementsByClassName('queue-thumb'));
    thumbs.forEach(function (thumb) {
      thumb.style.visibility = 'hidden';
    });
    document.getElementById('preview-text').style.visibility = 'visible';
    document.getElementById('thumb-interpolation').style.visibility = 'visible';
    document.getElementById('timecode').style.visibility = 'visible';
    document.getElementById('framecount').style.visibility = 'visible';
    document.getElementById('preview').style.visibility = 'visible';
  }
}
queueBtn.addEventListener('click', toggleQueueTab);

document.getElementById('clear-queue-btn').addEventListener('click', () => {
  queue.length = 0;
  renderQueueItem()
  document.getElementById('clear-queue-btn').style.visibility = 'hidden';
  sessionStorage.setItem('queueLength', queue.length);
  let empty = document.createElement('span');
  empty.setAttribute('id', 'queue-blank');
  empty.textContent = 'No items scheduled for processing, add items to the queue to get started.';
  queueTab.append(empty);
  enhancr.terminal('Cleared queue.')
})


