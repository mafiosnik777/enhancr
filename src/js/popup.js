const shell = require("electron").shell;
const axios = require('axios');

const fs = require('fs-extra');
const path = require('path');

const os = require('os');

const { clipboard } = require('electron')

const { spawn } = require('child_process');

const { ipcRenderer } = require("electron");

var terminal = document.getElementById("terminal-text");

const remote = require('@electron/remote');
const { BrowserWindow } = remote;
  
let win = BrowserWindow.getFocusedWindow();

var playModalBtn = document.getElementById("play-modal-label");
var pathModalBtn = document.getElementById("path-modal-label");
var shareModalBtn = document.getElementById("share-modal-label");

var closeModal = document.getElementById("close-success");

var progressSpan = document.getElementById('progress-span');

function playBackVideo() {
    shell.openPath(sessionStorage.getItem("pipeOutPath"));
}

function openVideoPath() {
    shell.showItemInFolder(sessionStorage.getItem("pipeOutPath"));
}

function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time));
}

function getTmpPath() {
    if (process.platform == 'win32') {
      return os.tmpdir() + "\\enhancr\\";
    } else {
      return os.tmpdir() + "/enhancr/";
    }
  }
  
var tempPath = getTmpPath();

async function shareVideo() {
    closeModal.click();

    const response = await axios.get('https://api.gofile.io/getServer');
    var bestServer = response.data.data.server;

    var pipeOutPath = sessionStorage.getItem('pipeOutPath');
    var fpsMeter = document.getElementById('fps-meter');
    fpsMeter.style.visibility = "hidden";
    var eta = document.getElementById('eta');
    eta.style.visibility = "hidden";

    const progressDone = document.getElementById("progress-done");

    sessionStorage.setItem('uploadStatus', 'uploading');
    let cmd = `curl -F file=@"${pipeOutPath}" https://${bestServer}.gofile.io/uploadFile -o ${path.join(tempPath, '/upload.json')}`;
    let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'] });
    process.stdout.write('');
    term.stdout.on('data', (data) => {
        sessionStorage.setItem("uploadProgress", data);
    });
    term.stderr.on('data', (data) => {
        sessionStorage.setItem("uploadProgress", data);
    });
    term.on("close", () => {
        win.setProgressBar(-1, {mode: "none"});
        win.flashFrame(true);
        sessionStorage.setItem('uploadStatus', 'done');
        const json = JSON.parse(fs.readFileSync(path.join(tempPath, '/upload.json')));
        terminal.textContent += "\r\n[gofile.io] Upload completed: " + json.data.downloadPage;
        terminal.scrollTop = terminal.scrollHeight;
        progressSpan.textContent = `Upload complete: ${path.basename(pipeOutPath)} | 100%`;
        var notification = new Notification("Upload completed", { icon: "", body: json.data.downloadPage });
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
            progressSpan.textContent = `Uploading ${path.basename(pipeOutPath)} | ${percentage}%`;
            progressDone.style.width = `${percentage}%`;
            if (percentage == 100) {
                sessionStorage.setItem('uploadStatus', 'done');
            }
            win.setProgressBar(parseInt(percentage) / 100);
            ipcRenderer.send('rpc-uploading', parseInt(percentage), path.basename(pipeOutPath));
            if (sessionStorage.getItem('uploadStatus') == 'done') {
                win.setProgressBar(-1);
                win.flashFrame(true);
                clearInterval(progressInterval);
            }
        } else {
            // do nothing
        }
    }, 1000)
};

playModalBtn.addEventListener("click", playBackVideo);
pathModalBtn.addEventListener("click", openVideoPath);
shareModalBtn.addEventListener("click", shareVideo);