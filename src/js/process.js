const path = require("path");
const { ipcRenderer } = require("electron");
const fs = require("fs-extra");

const remote = require('@electron/remote');
const { BrowserWindow } = remote;
let win = BrowserWindow.getFocusedWindow();

const thumbInterpolation = document.getElementById("thumb-interpolation");
const thumbUpscaling = document.getElementById("thumb-upscaling");
const thumbRestoration = document.getElementById("thumb-restoration");

const processOverlay = document.getElementById("process-overlay");
const progressSpan = document.getElementById("progress-span");
const progressDone = document.getElementById("progress-done");

const framecountCurrent = document.getElementById("framecount-current");
const framecountTotal = document.getElementById("framecount-total");
const timecodeCurrent = document.getElementById("timecode-current");
const timecodeDuration = document.getElementById("timecode-duration");

const preview = document.getElementById("preview");

sessionStorage.setItem('stopFlag', 'false');

class Process {
    static startProcessRestoration() {
        thumbRestoration.style.visibility = "hidden";
        if (sessionStorage.getItem('status') != 'error') {
            processOverlay.style.visibility = "visible";
        }
        var restoreInterval = setInterval(function () {
            var terminal = document.getElementById("terminal-text");
            sessionStorage.setItem("terminalScrolling", "true");
            terminal.scrollTop = terminal.scrollHeight;
            var progress = sessionStorage.getItem("progress");
            var status = sessionStorage.getItem("status");
            try {
                if (status == "done") {
                    console.log("Completed restoring, interval has been cleared.");
                    var filename = path.basename(sessionStorage.getItem('currentFile'));
                    if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                    progressSpan.textContent = filename + " | Complete"
                    console.log("Tmp folder cleared.");
                    win.setProgressBar(-1, {
                        mode: "none"
                    });
                    win.flashFrame(true);
                    progressDone.style.width = "100%";
                    let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                    let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                    queueProgress.innerHTML = '100%';
                    let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                    queueProgressOverlay.style.width = '100%';
                    let queueIcon = document.getElementById(`queue-item-remove${currentIndex}`);
                    queueIcon.removeAttribute("class");
                    queueIcon.classList.add('fa-solid');
                    if (sessionStorage.getItem('stopFlag') == 'false') {
                        queueIcon.classList.add('fa-check');
                        queueIcon.style.color = '#50C878';
                    } else {
                        queueIcon.classList.add('fa-xmark');
                        queueIcon.style.color = '#ca3433';
                        sessionStorage.setItem('stopFlag', 'false');
                    }
                    queueIcon.classList.add('queue-item-done');
                    preview.src = "";
                    timecodeCurrent.textContent = "00:00"
                    timecodeDuration.textContent = "00:00"
                    framecountCurrent.textContent = "0"
                    framecountTotal.textContent = "???"
                    processOverlay.style.visibility = "hidden";
                    thumbRestoration.style.visibility = "visible";
                    if (document.getElementById('queue').style.visibility == 'hidden') {
                        document.getElementById('top-bar-queue').click();
                        document.getElementById('preview-text').style.visibility = 'hidden';
                        document.getElementById('timecode').style.visibility = 'hidden';
                        document.getElementById('framecount').style.visibility = 'hidden';
                        document.getElementById('preview').style.visibility = 'hidden';
                    } else {
                        thumbInterpolation.style.visibility = "hidden";
                        thumbUpscaling.style.visibility = "hidden";
                        thumbRestoration.style.visibility = "hidden"
                    }
                    clearInterval(restoreInterval);
                }
            } catch {
                // ignore
            }
            try {
                if (progress.startsWith("Frame:") && progress !== null) {
                    var cleaned = progress.replace("Frame: ", "");
                    var currentFrame = cleaned.split("/")[0];
                    sessionStorage.setItem('currentFrame', currentFrame);
                    var totalFrames = cleaned.split("/")[1].split(" (")[0];
                    sessionStorage.setItem('totalFrames', totalFrames);
                    var percentage = Math.ceil(100 * (currentFrame / totalFrames));
                    sessionStorage.setItem("percent", percentage + "%");
                    progressDone.style.width = percentage + "%";

                    let queueIndex = Number(sessionStorage.getItem('queueIndex'));
                    sessionStorage.setItem(`queuePercent${queueIndex}`, queueIndex || 0);

                    win.setProgressBar(parseInt(percentage) / 100);
                    let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                    let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                    queueProgress.innerHTML = percentage + '%';
                    let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                    queueProgressOverlay.style.width = percentage + '%';
                    var filename = path.basename(sessionStorage.getItem('currentFile'));
                    if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                    progressSpan.textContent =
                        filename + " | " + percentage + "%";
                    const fpsMeter = document.getElementById("fps-meter");
                    const fpsMeterText = document.getElementById("fps-text");
                    const etaElement = document.getElementById("eta");
                    const etaText = document.getElementById("eta-text");
                    if (progress.includes("fps")) {
                        var fps = progress.split("(")[1].split(")")[0];
                        fpsMeter.style.display = "inline-block";
                        fpsMeterText.textContent = " " + fps;
                        var engineCleaned = sessionStorage.getItem('engine').split("- ")[1];
                        var fpsNumber = parseFloat(fps.replace(" fps"));
                        var eta = Math.round((totalFrames - currentFrame) / fpsNumber / 60);
                        etaElement.style.display = "inline-block";
                        etaText.textContent = " " + eta + " minutes";
                        var engineCleaned = sessionStorage.getItem('engine').split("- ")[1];
                        ipcRenderer.send('rpc-restoration', fpsNumber, engineCleaned, percentage);
                    }
                }
            } catch (error) {
                // ignore
            }
        }, 1000);
    }

    static startProcessUpscaling() {
        thumbInterpolation.style.visibility = "hidden";
        thumbUpscaling.style.visibility = "hidden";
        if (sessionStorage.getItem('status') != 'error') {
            processOverlay.style.visibility = "visible";
        }
        var upscaleInterval = setInterval(function () {
            var terminal = document.getElementById("terminal-text");
            sessionStorage.setItem("terminalScrolling", "true");
            terminal.scrollTop = terminal.scrollHeight;
            var progress = sessionStorage.getItem("progress");
            var status = sessionStorage.getItem("status");
            try {
                if (status == "done") {
                    console.log("Completed upscaling, interval has been cleared.");
                    var filename = path.basename(sessionStorage.getItem('currentFile'));
                    if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                    progressSpan.textContent = filename + " | Complete"
                    console.log("Tmp folder cleared.");
                    win.setProgressBar(-1, {
                        mode: "none"
                    });
                    win.flashFrame(true);
                    progressDone.style.width = "100%";
                    let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                    let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                    queueProgress.innerHTML = '100%';
                    let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                    queueProgressOverlay.style.width = '100%';
                    let queueIcon = document.getElementById(`queue-item-remove${currentIndex}`);
                    queueIcon.removeAttribute("class");
                    queueIcon.classList.add('fa-solid');
                    if (sessionStorage.getItem('stopFlag') == 'false') {
                        queueIcon.classList.add('fa-check');
                        queueIcon.style.color = '#50C878';
                    } else {
                        queueIcon.classList.add('fa-xmark');
                        queueIcon.style.color = '#ca3433';
                        sessionStorage.setItem('stopFlag', 'false');
                    }
                    queueIcon.classList.add('queue-item-done');
                    preview.src = "";
                    timecodeCurrent.textContent = "00:00"
                    timecodeDuration.textContent = "00:00"
                    framecountCurrent.textContent = "0"
                    framecountTotal.textContent = "???"
                    processOverlay.style.visibility = "hidden";
                    thumbInterpolation.style.visibility = "hidden";
                    thumbUpscaling.style.visibility = "visible";
                    if (document.getElementById('queue').style.visibility == 'hidden') {
                        document.getElementById('top-bar-queue').click();
                        document.getElementById('preview-text').style.visibility = 'hidden';
                        document.getElementById('timecode').style.visibility = 'hidden';
                        document.getElementById('framecount').style.visibility = 'hidden';
                        document.getElementById('preview').style.visibility = 'hidden';
                    } else {
                        thumbInterpolation.style.visibility = "hidden";
                        thumbUpscaling.style.visibility = "hidden";
                        thumbRestoration.style.visibility = "hidden"
                    }
                    clearInterval(upscaleInterval);
                }
            } catch {
                // ignore
            }
            try {
                if (progress.startsWith("Frame:") && progress !== null) {
                    var cleaned = progress.replace("Frame: ", "");
                    var currentFrame = cleaned.split("/")[0];
                    sessionStorage.setItem('currentFrame', currentFrame);
                    var totalFrames = cleaned.split("/")[1].split(" (")[0];
                    sessionStorage.setItem('totalFrames', totalFrames);
                    var percentage = Math.ceil(100 * (currentFrame / totalFrames));
                    sessionStorage.setItem("percent", percentage + "%");
                    progressDone.style.width = percentage + "%";

                    let queueIndex = Number(sessionStorage.getItem('queueIndex'));
                    sessionStorage.setItem(`queuePercent${queueIndex}`, queueIndex || 0);

                    win.setProgressBar(parseInt(percentage) / 100);
                    let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                    let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                    queueProgress.innerHTML = percentage + '%';
                    let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                    queueProgressOverlay.style.width = percentage + '%';
                    var filename = path.basename(sessionStorage.getItem('currentFile'));
                    if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                    progressSpan.textContent =
                        filename + " | " + percentage + "%";
                    const fpsMeter = document.getElementById("fps-meter");
                    const fpsMeterText = document.getElementById("fps-text");
                    const etaElement = document.getElementById("eta");
                    const etaText = document.getElementById("eta-text");
                    if (progress.includes("fps")) {
                        var fps = progress.split("(")[1].split(")")[0];
                        fpsMeter.style.display = "inline-block";
                        fpsMeterText.textContent = " " + fps;
                        var fpsNumber = parseFloat(fps.replace(" fps"));
                        var eta = Math.round((totalFrames - currentFrame) / fpsNumber / 60);
                        etaElement.style.display = "inline-block";
                        etaText.textContent = " " + eta + " minutes";
                        var engineCleaned = sessionStorage.getItem('engine').split("- ")[1];
                        ipcRenderer.send('rpc-upscaling', fpsNumber, engineCleaned, percentage);
                    }
                }
            } catch (error) {
                // ignore
            }
        }, 1000);
    }

    static startProcessInterpolation() {
        thumbInterpolation.style.visibility = "hidden";
        thumbUpscaling.style.visibility = "hidden";
        thumbRestoration.style.visibility = "hidden";
        if (sessionStorage.getItem('status') != 'error') {
            processOverlay.style.visibility = "visible";
        }
        let videoFrameRateOut = document.getElementById('framerate').innerHTML.split('➜ ')[1];
        if (sessionStorage.getItem('status') == "upscaling" || sessionStorage.getItem('status') == "interpolation" || sessionStorage.getItem('status') == "restoring") {
            console.log('process already running');
        } else {
            var interpolationInterval = setInterval(function () {
                var terminal = document.getElementById("terminal-text");
                sessionStorage.setItem("terminalScrolling", "true");
                terminal.scrollTop = terminal.scrollHeight;
                var progress = sessionStorage.getItem("progress");
                var status = sessionStorage.getItem("status");
                try {
                    if (status !== null && status == "done") {
                        console.log("Completed interpolation, interval has been cleared.");
                        var filename = path.basename(sessionStorage.getItem('currentFile'));
                        if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                        progressSpan.textContent = filename + " | Complete"
                        console.log("Tmp folder cleared.");
                        win.setProgressBar(-1, {
                            mode: "none"
                        });
                        win.flashFrame(true);
                        progressDone.style.width = "100%";
                        let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                        let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                        queueProgress.innerHTML = '100%';
                        let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                        queueProgressOverlay.style.width = '100%';
                        let queueIcon = document.getElementById(`queue-item-remove${currentIndex}`);
                        queueIcon.removeAttribute("class");
                        queueIcon.classList.add('fa-solid');
                        if (sessionStorage.getItem('stopFlag') == 'false') {
                            queueIcon.classList.add('fa-check');
                            queueIcon.style.color = '#50C878';
                        } else {
                            queueIcon.classList.add('fa-xmark');
                            queueIcon.style.color = '#ca3433';
                            sessionStorage.setItem('stopFlag', 'false');
                        }
                        queueIcon.classList.add('queue-item-done');
                        preview.src = "";
                        timecodeCurrent.textContent = "00:00"
                        timecodeDuration.textContent = "00:00"
                        framecountCurrent.textContent = "0"
                        framecountTotal.textContent = "???"
                        processOverlay.style.visibility = "hidden";
                        if (document.getElementById('queue').style.visibility == 'hidden') {
                            document.getElementById('top-bar-queue').click();
                            document.getElementById('preview-text').style.visibility = 'hidden';
                            document.getElementById('timecode').style.visibility = 'hidden';
                            document.getElementById('framecount').style.visibility = 'hidden';
                            document.getElementById('preview').style.visibility = 'hidden';
                        } else {
                            thumbInterpolation.style.visibility = "hidden";
                            thumbUpscaling.style.visibility = "hidden";
                            thumbRestoration.style.visibility = "hidden"
                        }
                        clearInterval(interpolationInterval);
                    }
                } catch {
                    // ignore
                }
                try {
                    if (progress.startsWith("Frame:") && progress !== null) {
                        var cleaned = progress.replace("Frame: ", "");
                        var currentFrame = cleaned.split("/")[0];
                        sessionStorage.setItem('currentFrame', currentFrame);
                        var totalFrames = cleaned.split("/")[1].split(" (")[0];
                        sessionStorage.setItem('totalFrames', totalFrames);
                        var percentage = Math.ceil(100 * (currentFrame / totalFrames));
                        sessionStorage.setItem("percent", percentage + "%");
                        progressDone.style.width = percentage + "%";

                        let queueIndex = Number(sessionStorage.getItem('queueIndex'));
                        sessionStorage.setItem(`queuePercent${queueIndex}`, percentage || 0);

                        win.setProgressBar(parseInt(percentage) / 100);
                        let currentIndex = parseInt(sessionStorage.getItem('queueIndex'));
                        let queueProgress = document.getElementById(`queue-percent${currentIndex}`);
                        queueProgress.innerHTML = percentage + '%';
                        let queueProgressOverlay = document.getElementById(`queue-item-progress-overlay${currentIndex}`);
                        queueProgressOverlay.style.width = percentage + '%';
                        var filename = path.basename(sessionStorage.getItem('currentFile'));
                        if (filename.length > 60) filename = filename.substr(0, 60) + "...";
                        progressSpan.textContent =
                            filename + " | " + percentage + "%";
                        const fpsMeter = document.getElementById("fps-meter");
                        const fpsMeterText = document.getElementById("fps-text");
                        const etaElement = document.getElementById("eta");
                        const etaText = document.getElementById("eta-text");
                        if (progress.includes("fps")) {
                            var fps = progress.split("(")[1].split(")")[0];
                            fpsMeter.style.display = "inline-block";
                            fpsMeterText.textContent = " " + fps;
                            // adjust preview speed based on fps
                            let playbackSpeed = (parseFloat(fps) / parseFloat(videoFrameRateOut)).toFixed(2);
                            if (playbackSpeed > 1) {
                                preview.playbackRate = playbackSpeed;
                            }
                            var fpsNumber = parseFloat(fps.replace(" fps"));
                            var eta = Math.round((totalFrames - currentFrame) / fpsNumber / 60);
                            etaElement.style.display = "inline-block";
                            etaText.textContent = " " + eta + " minutes";
                            var engineCleaned = sessionStorage.getItem('engine').split("- ")[1];
                            ipcRenderer.send('rpc-interpolation', fpsNumber, engineCleaned, percentage);
                        }
                    }
                } catch {
                    // ignore
                }
            }, 1000);
        }
    }
}

module.exports = Process;

