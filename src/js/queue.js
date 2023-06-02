const { PromisePool } = require('@supercharge/promise-pool')
const path = require("path");
const remote = require('@electron/remote');

let axios;

if (remote.app.isPackaged == false) {
  axios = require("axios");
} else {
  axios = require(path.join(process.resourcesPath, '/app.asar.unpacked/node_modules/axios/dist/browser/axios.cjs'))
}

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

let ffmpegPath;
let ffprobePath;

if (remote.app.isPackaged == false) {
  ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
  ffprobePath = require('@ffprobe-installer/ffprobe').path;
} else {
  ffmpegPath = require('@ffmpeg-installer/ffmpeg').path.replace('app.asar', 'app.asar.unpacked');
  ffprobePath = require('@ffprobe-installer/ffprobe').path.replace('app.asar', 'app.asar.unpacked');
}

ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);

let queueTab = document.getElementById('queue');
let queueBtn = document.getElementById('top-bar-queue');

var progressSpan = document.getElementById('progress-span');

var terminal = document.getElementById("terminal-text");

const exportBtn = document.getElementById('export-btn');
const realtimeBtn = document.getElementById('realtime-btn');

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
      const cacheBuster = new Date().getTime();
      let thumb = document.getElementById(`queue-thumb${i}`);
      thumb.setAttribute('src', path.join(appDataPath, `/.enhancr/thumbs/queue${i}.png?${cacheBuster}`));
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
      let percentValue = sessionStorage.getItem(`queuePercent${i}`);
      percent.innerHTML = `${percentValue ? percentValue : '0'}%`;
      // progressOverlay.style.width = `${percentValue ? percentValue : '0'}%`;
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
    if (queueItem.status == '1') play.innerHTML = "<i class='fa-solid fa-film'></i> Play video"; else play.innerHTML = "<i class='fa-solid fa-link'></i> Chain";
    let openPath = document.createElement('button');
    openPath.classList.add('menu__item');
    openPath.classList.add('context-item1');
    openPath.setAttribute('id', `context-item1-queue${i}`);
    listItem2.append(openPath);
    if (queueItem.status == '1') openPath.innerHTML = "<i class='fa-solid fa-folder-closed'></i> Open in folder"; else openPath.innerHTML = "<i class='fa-solid fa-pencil'></i> Output name";
    let share = document.createElement('button');
    share.classList.add('menu__item');
    share.classList.add('context-item2');
    share.setAttribute('id', `context-item2-queue${i}`);
    listItem3.append(share);
    if (queueItem.status == '1') share.innerHTML = '<i class="fa-solid fa-share-nodes"></i> Share'; else share.innerHTML = "<i class='fa-solid fa-scissors'></i> Trim";
    // create progress overlay
    let progressOverlay = document.createElement('div');
    progressOverlay.classList.add('queue-item-progress-overlay');
    progressOverlay.setAttribute('id', `queue-item-progress-overlay${i}`);
    progressOverlay.style.background = ThemeEngine.getTheme(sessionStorage.getItem('theme'));
    if (queueItem.status == '1') {
      let percentValue = sessionStorage.getItem(`queuePercent${i}`);
      percent.innerHTML = `${percentValue ? percentValue : '0'}%`;
      progressOverlay.style.width = `${percentValue ? percentValue : '0'}%`;
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
    const modelMap = {
      'Channel Attention - CAIN (NCNN)': 'CAIN',
      'Optical Flow - RIFE (NCNN)': 'RIFE',
      'Optical Flow - RIFE (TensorRT)': 'RIFE',
      'Channel Attention - CAIN (TensorRT)': 'CAIN',
      'Channel Attention - CAIN (DirectML)': 'CAIN',
      'GMFlow - GMFSS (PyTorch)': 'GMFSS',
      'GMFlow - GMFSS (TensorRT)': 'GMFSS',
      'Upscaling - RealESRGAN (TensorRT)': 'RealESRGAN',
      'Upscaling - RealESRGAN (NCNN)': 'RealESRGAN',
      'Upscaling - RealESRGAN (DirectML)': 'RealESRGAN',
      'Upscaling - ShuffleCUGAN (TensorRT)': 'ShuffleCUGAN',
      'Upscaling - ShuffleCUGAN (NCNN)': 'ShuffleCUGAN',
      'Upscaling - RealCUGAN (TensorRT)': 'RealCUGAN',
      'Upscaling - SwinIR (TensorRT)': 'SwinIR',
      'Restoration - DPIR (TensorRT)': 'DPIR',
      'Restoration - DPIR (DirectML)': 'DPIR',
      'Restoration - ScuNET (TensorRT)': 'ScuNET',
      'Restoration - RealESRGAN (1x) (NCNN)': 'RealESRGAN',
      'Restoration - RealESRGAN (1x) (TensorRT)': 'RealESRGAN',
      'Restoration - RealESRGAN (1x) (DirectML)': 'RealESRGAN',
    };

    //chaining / play video
    const contextItem0 = Array.from(document.getElementsByClassName('context-item0'));
    contextItem0.forEach((item) => {
      if (!item.classList.contains('listener')) {
        item.classList.add('listener');
        item.addEventListener('click', () => {
          const id = item.id;
          const index = id.charAt(id.length - 1);
          const chainModal = document.getElementById('modal-chain');
          const mode = queue[index].mode;
          
          if (item.innerHTML === '<i class="fa-solid fa-film"></i> Play video') {
            let model;
            let outputPath;
    
            if (sessionStorage.getItem(`out${index}`) !== null) {
              outputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
              shell.openPath(outputPath);
            } else {
              if (mode === 'interpolation') {
                model = modelMap[queue[index].engine] || '';
                outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
              } else if (mode === 'upscaling') {
                model = modelMap[queue[index].engine] || '';
                outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
              } else if (mode === 'restoration') {
                model = modelMap[queue[index].engine] || 'RealESRGAN';
                outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
              }
              shell.openPath(outputPath);
            }
          } else {
            const chainBtnInterpolation = document.getElementById('chain-interpolation');
            const chainBtnUpscaling = document.getElementById('chain-upscaling');
            const chainBtnRestoration = document.getElementById('chain-restoration');
    
            if (mode === 'interpolation') {
              chainBtnInterpolation.style.display = 'none';
              chainBtnUpscaling.style.display = 'block';
              chainBtnRestoration.style.display = 'block';
            } else if (mode === 'upscaling') {
              chainBtnInterpolation.style.display = 'block';
              chainBtnUpscaling.style.display = 'none';
              chainBtnRestoration.style.display = 'block';
            } else {
              chainBtnInterpolation.style.display = 'block';
              chainBtnUpscaling.style.display = 'block';
              chainBtnRestoration.style.display = 'none';
            }
    
            openModal(chainModal);
    
            const closeModalAndSetInputPath = (videoInputPath, inputTextElement, tabBtn) => {
              closeModal(chainModal);
              sessionStorage.setItem("inputPath", videoInputPath);
    
              if (videoInputPath.length >= 55 && path.basename(videoInputPath).length >= 55) {
                inputTextElement.textContent = "../" + path.basename(videoInputPath).substr(0, 55) + "\u2026";
              } else if (videoInputPath.length >= 55) {
                inputTextElement.textContent = "../" + path.basename(videoInputPath);
              } else {
                inputTextElement.textContent = videoInputPath;
              }
    
              tabBtn.click();
            };
    
            if (!chainBtnUpscaling.classList.contains('listener')) {
              chainBtnUpscaling.classList.add('listener');
              chainBtnUpscaling.addEventListener('click', () => {
                const upscaleTabBtn = document.getElementById('upscale-side-arrow');
                let videoInputPath;
                let model;
    
                if (sessionStorage.getItem(`out${index}`) === null) {
                  if (mode === 'interpolation') {
                    model = modelMap[queue[index].engine] || '';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
                  } else {
                    model = modelMap[queue[index].engine] || 'RealESRGAN';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
    
                sessionStorage.setItem("upscaleInputPath", videoInputPath);
                closeModalAndSetInputPath(videoInputPath, document.getElementById("upscale-input-text"), upscaleTabBtn);
              });
            }
    
            if (!chainBtnRestoration.classList.contains('listener')) {
              chainBtnRestoration.classList.add('listener');
              chainBtnRestoration.addEventListener('click', () => {
                const restoreTabBtn = document.getElementById('restore-side');
                let videoInputPath;
                let model;
    
                if (sessionStorage.getItem(`out${index}`) === null) {
                  if (mode === 'interpolation') {
                    model = modelMap[queue[index].engine] || '';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
                  } else {
                    model = modelMap[queue[index].engine] || '';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
    
                sessionStorage.setItem("inputPathRestore", videoInputPath);
                closeModalAndSetInputPath(videoInputPath, document.getElementById("restore-input-text"), restoreTabBtn);
              });
            }
    
            if (!chainBtnInterpolation.classList.contains('listener')) {
              chainBtnInterpolation.classList.add('listener');
              chainBtnInterpolation.addEventListener('click', () => {
                const interpolationTabBtn = document.getElementById('interpolate-side');
                let videoInputPath;
                let model;
    
                if (sessionStorage.getItem(`out${index}`) === null) {
                  if (mode === 'upscaling') {
                    model = modelMap[queue[index].engine] || '';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
                  } else {
                    model = modelMap[queue[index].engine] || 'RealESRGAN';
                    videoInputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
                  }
                } else {
                  videoInputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
                }
    
                sessionStorage.setItem("inputPath", videoInputPath);
                closeModalAndSetInputPath(videoInputPath, document.getElementById("input-video-text"), interpolationTabBtn);
              });
            }
          }
        });
      }
    });

    // edit export name / show in folder
    const contextItem1 = Array.from(document.getElementsByClassName('context-item1'));

    contextItem1.forEach(item => {
      if (!item.classList.contains('listener')) {
        item.classList.add('listener');
        item.addEventListener('click', () => {
          const id = item.id;
          const index = id.charAt(id.length - 1);

          if (item.innerHTML === '<i class="fa-solid fa-folder-closed"></i> Open in folder') {
            let outputPath;
            const queueItem = queue[index];

            if (sessionStorage.getItem(`out${index}`) !== null) {
              outputPath = path.join(queueItem.output, sessionStorage.getItem(`out${index}`) + queueItem.extension);
              remote.shell.showItemInFolder(outputPath);
            } else if (queueItem.mode === 'interpolation') {
              const engine = queueItem.engine;
              const model = modelMap[engine];
              outputPath = path.join(queueItem.output, path.parse(queueItem.file).name + `_${model}-2x${queueItem.extension}`);
              remote.shell.showItemInFolder(outputPath);
            } else if (queueItem.mode === 'upscaling') {
              const engine = queueItem.engine;
              const model = modelMap[engine];
              outputPath = path.join(queueItem.output, path.parse(queueItem.file).name + `_${model}-${queueItem.scale}x${queueItem.extension}`);
              console.log(outputPath);
              remote.shell.showItemInFolder(outputPath);
            } else if (queueItem.mode === 'restoration') {
              const engine = queueItem.engine;
              const model = engine.includes("RealESRGAN") ? "RealESRGAN" : "DPIR";
              outputPath = path.join(queueItem.output, path.parse(queueItem.file).name + `_${model}-1x${queueItem.extension}`);
              remote.shell.showItemInFolder(outputPath);
            }
          } else {
            const editModal = document.getElementById('modal-edit');
            openModal(editModal);
            const editField = document.getElementById('edit-export-name');
            editField.value = '';
            const fileName = document.getElementById(`queue-title${index}`);
            editField.placeholder = path.parse(fileName.innerHTML).name;
            const saveBtn = document.getElementById('save-edit-btn');
            saveBtn.addEventListener('click', () => {
              if (editField.value !== '') {
                sessionStorage.setItem(`out${index}`, editField.value);
                closeModal(editModal);
                enhancr.terminal(`Set output file name for position '${index}' in queue to: '${editField.value}'`);
              }
            });
          }
        });
      }
    });

  // open folder / share
  const contextItem2 = Array.from(document.getElementsByClassName('context-item2'));

  contextItem2.forEach((item) => {
    if (!item.classList.contains('listener')) {
      item.classList.add('listener');
      item.addEventListener('click', async () => {
        const id = item.id;
        const index = id.charAt(id.length - 1);
        if (item.innerHTML === '<i class="fa-solid fa-share-nodes"></i> Share') {
          const response = await axios.get('https://api.gofile.io/getServer');
          const bestServer = response.data.data.server;
          let outputPath;
          let model;

          if (sessionStorage.getItem(`out${index}`) !== null) {
            outputPath = path.join(queue[index].output, sessionStorage.getItem(`out${index}`) + queue[index].extension);
          } else if (queue[index].mode === 'interpolation') {
            model = modelMap[queue[index].engine];
            outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-2x${queue[index].extension}`);
          } else if (queue[index].mode === 'upscaling') {
            model = modelMap[queue[index].engine];
            outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-${queue[index].scale}x${queue[index].extension}`);
          } else if (queue[index].mode === 'restoration') {
            model = modelMap[queue[index].engine] || "RealESRGAN";
            outputPath = path.join(queue[index].output, path.parse(queue[index].file).name + `_${model}-1x${queue[index].extension}`);
          }

          const progressDone = document.getElementById("progress-done");
          sessionStorage.setItem('uploadStatus', 'uploading');
          const cmd = `curl -F file=@"${outputPath}" https://${bestServer}.gofile.io/uploadFile -o ${path.join(tempPath, '/upload.json')}`;
          const term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'] });

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
            const notification = new Notification("Upload completed", { icon: "./assets/enhancr.png", body: json.data.downloadPage });
            progressDone.style.width = `100%`;
            clipboard.writeText(json.data.downloadPage);
            terminal.textContent += "\r\n[gofile.io] Copied link to clipboard.";
            window.open(json.data.downloadPage, "_blank");
          });

          const progressInterval = setInterval(() => {
            const progress = sessionStorage.getItem('uploadProgress');
            if (progress.length >= 22) {
              const trimmed = progress.trim();
              const percentage = trimmed.split(' ')[0];
              terminal.textContent += `\r\n[gofile.io] Uploading... (${percentage}%)`;
              terminal.scrollTop = terminal.scrollHeight;
              progressSpan.textContent = `Uploading ${path.basename(outputPath)} | ${percentage}%`;
              progressDone.style.width = `${percentage}%`;
              if (percentage === 100) {
                sessionStorage.setItem('uploadStatus', 'done');
              }
              win.setProgressBar(parseInt(percentage) / 100);
              ipcRenderer.send('rpc-uploading', parseInt(percentage), path.basename(outputPath));
              if (sessionStorage.getItem('uploadStatus') === 'done') {
                win.setProgressBar(-1);
                win.flashFrame(true);
                clearInterval(progressInterval);
              }
            }
          }, 1000);
        } else {
          const trimModal = document.getElementById('modal-trim');
          openModal(trimModal);
          const saveTimestamps = document.getElementById('save-timestamps-btn');
          saveTimestamps.addEventListener('click', () => {
            sessionStorage.setItem(`trim${index}`, document.getElementById('timestamp-trim').value);
            closeModal(trimModal);
          });
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
    for (var i = 0; i < file.length; i++) {
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
    for (var i = 0; i < file.length; i++) {
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
    for (var i = 0; i < file.length; i++) {
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
              realtimeBtn.style.pointerEvents = 'none';
              let realtimeBtnIcon = document.getElementById("realtime-btn-icon");
              realtimeBtnIcon.className = 'fa-solid fa-lock';
              realtimeBtnIcon.style.color = 'grey';
              exportBtn.innerHTML = '<span id="export-button"><i class="fa-solid fa-circle-stop" id="enhance-icon"></i></i> Stop processing video(s) <span id="key-shortcut">Ctrl + Enter</span></span>'
              document.getElementById('clear-queue-btn').style.visibility = 'hidden';
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

              let queueDots = document.getElementById(`queue-item-dots${queue.indexOf(item)}`);
              queueDots.style.pointerEvents = 'none';
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

          sessionStorage.removeItem('percent');
          sessionStorage.removeItem('currentFrame');
          sessionStorage.removeItem('totalFrames');
          sessionStorage.removeItem('trim${queue.indexOf(item)}');
          sessionStorage.removeItem('out${queue.indexOf(item)}');
          document.getElementById('progress-done').style.width = "0%";
          document.getElementById('eta').style.display = "none";
          document.getElementById('fps-meter').style.display = "none";
          document.getElementById('loading').style.display = "none";
          let queueDots = document.getElementById(`queue-item-dots${queue.indexOf(item)}`);
          queueDots.style.pointerEvents = 'auto';
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
          realtimeBtn.style.pointerEvents = 'auto';
          let realtimeBtnIcon = document.getElementById("realtime-btn-icon");
          realtimeBtnIcon.className = 'fa-regular fa-circle-play';
          realtimeBtnIcon.style.color = 'white';
          exportBtn.innerHTML = '<i class="fa-solid fa-turn-up fa-rotate-90" id="enhance-icon"></i> Enhance video(s) <span id="key-shortcut">Ctrl + Enter</span>';
          document.getElementById('clear-queue-btn').style.visibility = 'visible';

          let errorCount = Number(sessionStorage.getItem("errorCount")) || 0;
          if (queue.length == 0) {
            document.getElementById('clear-queue-btn').style.visibility = 'hidden';
            enhancr.terminal('Queue is empty. Add media to queue to get started.\r\n');
          } else if (errorCount >= 0){
            enhancr.terminal(`Completed queue with ${errorCount} error(s).\r\n`);
            sessionStorage.setItem("errorCount", "0");
          } else {
            enhancr.terminal('Completed queue successfully.\r\n');
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
      find('name', 'trtexec', false).then(function (list) {
        var i;
        for (i = 0; i < list.length; i++) {
          process.kill(list[i].pid);
      }
      let killVsPipe = () => {
        find('name', 'VSPipe', false).then(function (list) {
          var i;
          for (i = 0; i < list.length; i++) {
            process.kill(list[i].pid);
          }
        })
      }
      setTimeout(killVsPipe, 1000);
      }).then(function () {
        realtimeBtn.style.pointerEvents = 'auto';
        let realtimeBtnIcon = document.getElementById("realtime-btn-icon");
        realtimeBtnIcon.className = 'fa-regular fa-circle-play';
        realtimeBtnIcon.style.color = 'white';

        exportBtn.innerHTML = '<i class="fa-solid fa-turn-up fa-rotate-90" id="enhance-icon"></i> Enhance video(s) <span id="key-shortcut">Ctrl + Enter</span>';
        enhancr.terminal('Stopped queue successfully.\r\n');
        setTimeout(() => {
          document.getElementById('progress-done').style.width = '0%';
          document.getElementById('progress-span').innerHTML = `${path.basename(sessionStorage.getItem('pipeOutPath'))} | Canceled`
          document.getElementById('loading').style.display = 'none';
          win.setProgressBar(-1, {
            mode: "none"
          });
          document.getElementById('eta').style.display = 'none';
          document.getElementById('fps-meter').style.display = 'none';
        }, 5000)
        ipcRenderer.send('rpc-done');
        terminal.scrollTop = terminal.scrollHeight;
        running = false;
      });
    }
  };
  exportBtn.addEventListener('click', processQueue);
  
  let realtimeIsRunning = false;

  let processRealtimeQueue = async () => {
    if (realtimeIsRunning) {
      return;
    }
    realtimeIsRunning = true;

    let initialPreviewState = document.getElementById('preview-check').checked;
    document.getElementById('preview-check').checked = false;
    sessionStorage.setItem('realtime', 'true');
    document.getElementById('preview-text').innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Realtime Playback active</span>'
    await processQueue();
    sessionStorage.setItem('realtime', 'false');
    document.getElementById('preview-text').innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Preview not initialized</span>'
    document.getElementById('preview-check').checked = initialPreviewState;
    realtimeIsRunning = false;
  }

  realtimeBtn.addEventListener('click', processRealtimeQueue);
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
    document.getElementById('timecode').style.visibility = 'visible';
    document.getElementById('framecount').style.visibility = 'visible';
    document.getElementById('preview').style.visibility = 'visible';

    if (sessionStorage.getItem('currentTab') == 'interpolation') document.getElementById('thumb-interpolation').style.visibility = 'visible';
    if (sessionStorage.getItem('currentTab') == 'upscaling') document.getElementById('thumb-upscaling').style.visibility = 'visible';
    if (sessionStorage.getItem('currentTab') == 'restoration') document.getElementById('thumb-restoration').style.visibility = 'visible';
  }
}
queueBtn.addEventListener('click', toggleQueueTab);

document.getElementById('clear-queue-btn').addEventListener('click', () => {
  queue.length = 0;
  renderQueueItem()
  document.getElementById('clear-queue-btn').style.visibility = 'hidden';
  sessionStorage.setItem('queueLength', queue.length);
  for(let i = 0; i < queue.length; i++) {
    try {
        sessionStorage.removeItem('queuePercent' + i);
        sessionStorage.removeItem('queueLength');
        sessionStorage.removeItem('percent');
        sessionStorage.remove('progress');
    } catch (error) {
    }
}
  let empty = document.createElement('span');
  empty.setAttribute('id', 'queue-blank');
  empty.textContent = 'No items scheduled for processing, add items to the queue to get started.';
  queueTab.append(empty);
  enhancr.terminal('Queue : Cleared')
})


