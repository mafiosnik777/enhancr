const { promises, fs } = require("fs");
const path = require("path");
const MediaInfoFactory = require("mediainfo.js");
const ratio = require('aspect-ratio')
const fse = require('fs-extra');

const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const ffprobePath = require('@ffprobe-installer/ffprobe').path;
ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);

var videoInput = document.getElementById("input-video-text");
var videoInputUpscale = document.getElementById("upscale-input-text")
var videoInputRestore = document.getElementById("restore-input-text")

var noMedia = document.getElementById("nomedia");
var mediaContainer = document.getElementById("media-container");
var mediaContainerUp = document.getElementById("media-container-upscale");
var mediaContainerRes = document.getElementById("media-container-restoration");

var mediaHeader = document.getElementById("header-text");
var infoFormat = document.getElementById("format");
var infoSize = document.getElementById("size");
var infoDimensions = document.getElementById("dimensions");
var infoFrameRate = document.getElementById("framerate");
var infoDuration = document.getElementById("duration");
var infoFramecount = document.getElementById("frame-count");
var infoBitrate = document.getElementById("bitrate");

var mediaHeaderUp = document.getElementById("header-text-upscale");
var infoFormatUp = document.getElementById("formatUp");
var infoSizeUp = document.getElementById("sizeUp");
var infoDimensionsUp = document.getElementById("dimensionsUp");
var infoFrameRateUp = document.getElementById("framerateUp");
var infoDurationUp = document.getElementById("durationUp");
var infoFramecountUp = document.getElementById("frame-countUp");
var infoBitrateUp = document.getElementById("bitrateUp");

var mediaHeaderRes = document.getElementById("header-text-restore");
var infoFormatRes = document.getElementById("formatRes");
var infoSizeRes = document.getElementById("sizeRes");
var infoDimensionsRes = document.getElementById("dimensionsRes");
var infoFrameRateRes = document.getElementById("framerateRes");
var infoDurationRes = document.getElementById("durationRes");
var infoFramecountRes = document.getElementById("frame-countRes");
var infoBitrateRes = document.getElementById("bitrateRes");

var previewTotalFrames = document.getElementById("framecount-total");
var timecodeDuration = document.getElementById("timecode-duration");

// observe video input changes (interpolation)
var MutationObserver = window.MutationObserver;

var observer = new MutationObserver(fetchMetadata);

observer.observe(videoInput, {
    childList: true
});

// observe video input changes (upscaling)
var MutationObserver = window.MutationObserver;

var observerUp = new MutationObserver(fetchMetadataUpscale);

observerUp.observe(videoInputUpscale, {
    childList: true
});

// observe scale changes (upscaling)
var MutationObserver = window.MutationObserver;

var observerScale = new MutationObserver(calcRes);

var scaleInput = document.getElementById("factor-span")
observerScale.observe(scaleInput, {
    childList: true
});

// observe video input changes (restoration)
var MutationObserver = window.MutationObserver;

var observerRes = new MutationObserver(fetchMetadataRestore);

observerRes.observe(videoInputRestore, {
    childList: true
});


// helper functions
function getReadChunkFunction(fileHandle) {
    async function readChunk(size, offset) {
        const buffer = new Uint8Array(size);
        await fileHandle.read(buffer, 0, size, offset);
        return buffer;
    }

    return readChunk;
}

async function readMetadata(filepath) {
    const mediaInfo = await MediaInfoFactory({ format: "JSON", coverData: true });
    const fileHandle = await promises.open(filepath, "r");
    const fileSize = (await fileHandle.stat()).size;
    const readChunk = getReadChunkFunction(fileHandle);
    const result = await mediaInfo.analyzeData(() => fileSize, readChunk);
    return result;
}

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

// fetch metadata (interpolation)
function fetchMetadata() {
    let inputPath = sessionStorage.getItem('inputPath');
    readMetadata(inputPath).then((result) => {
        const data = JSON.parse(result);

        const video = data.media.track.find((item) => item["@type"] === "Video");
        const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

        const container = data.media.track[0].Format;
        const filesize = data.media.track[0].FileSize;
        const bitRate = data.media.track[0].OverallBitRate;

        const aspectRatio = ratio(parseInt(Height), parseInt(Width));

        noMedia.style.visibility = "hidden";
        mediaContainer.style.visibility = "visible";

        mediaHeader.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

        console.log(data);

        infoFormat.textContent = container + ' - ' + Format;
        infoSize.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
        infoDimensions.textContent = Width + " x " + Height + " (" + aspectRatio + ")";
        infoFrameRate.textContent = FrameRate + " ➜ " + parseFloat(FrameRate) * 2;
        infoDuration.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
        infoFramecount.textContent = FrameCount;
        infoBitrate.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
        webFrame.clearCache()
        ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbInterpolation.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
        var thumbInterpolation = document.getElementById('interpolation-thumb-img');
        function setThumbInterpolation() {
            thumbInterpolation.src = path.join(appDataPath, '/.enhancr/thumbs/thumbInterpolation.png?' + Math.random() * 1000);
        }
        setTimeout(setThumbInterpolation, 2000);
    })
};
var getFactor = parseInt(document.getElementById("factor-span").textContent.split("x")[0]);
sessionStorage.setItem("upscaleFactor", getFactor)

// fetch metadata (upscaling)
function fetchMetadataUpscale() {
    let inputPath = sessionStorage.getItem('upscaleInputPath');
    let upscaleFactor = parseInt(sessionStorage.getItem("upscaleFactor"));
    readMetadata(inputPath).then((result) => {
        const data = JSON.parse(result);

        const video = data.media.track.find((item) => item["@type"] === "Video");
        const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

        const container = data.media.track[0].Format;
        const filesize = data.media.track[0].FileSize;
        const bitRate = data.media.track[0].OverallBitRate;

        const aspectRatio = ratio(parseInt(Height), parseInt(Width));

        noMedia.style.visibility = "hidden";
        if (sessionStorage.getItem('currentTab') == 'upscaling') {
            mediaContainerUp.style.visibility = "visible";
        }

        mediaHeaderUp.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

        console.log(data);

        infoFormatUp.textContent = container + ' - ' + Format;
        infoSizeUp.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
        infoDimensionsUp.textContent = Width + " x " + Height + " ➜ " + parseInt(Width) * upscaleFactor + " x " + parseInt(Height) * upscaleFactor + " (" + aspectRatio + ")";
        infoFrameRateUp.textContent = FrameRate;
        infoDurationUp.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
        infoFramecountUp.textContent = FrameCount;
        infoBitrateUp.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
        webFrame.clearCache()
        ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbUpscaling.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
        var thumbUpscaling = document.getElementById('upscaling-thumb-img');
        function setThumbUpscaling() {
            thumbUpscaling.src = path.join(appDataPath, '/.enhancr/thumbs/thumbUpscaling.png?' + Math.random() * 1000);
        }
        setTimeout(setThumbUpscaling, 1000)
    })
};

// calculate new res on scale changes
function calcRes() {
    var inputPath = sessionStorage.getItem('upscaleInputPath');
    let upscaleFactor = parseInt(sessionStorage.getItem("upscaleFactor"));
    readMetadata(inputPath).then((result) => {
        const data = JSON.parse(result);

        const video = data.media.track.find((item) => item["@type"] === "Video");
        const { Width, Height } = video;
        const aspectRatio = ratio(parseInt(Height), parseInt(Width));

        infoDimensionsUp.textContent = Width + " x " + Height + " ➜ " + parseInt(Width) * upscaleFactor + " x " + parseInt(Height) * upscaleFactor + " (" + aspectRatio + ")";
    })
};

// fetch metadata (restoration)
function fetchMetadataRestore() {
    let inputPath = sessionStorage.getItem('inputPathRestore');
    readMetadata(inputPath).then((result) => {
        const data = JSON.parse(result);

        const video = data.media.track.find((item) => item["@type"] === "Video");
        const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

        const container = data.media.track[0].Format;
        const filesize = data.media.track[0].FileSize;
        const bitRate = data.media.track[0].OverallBitRate;

        const aspectRatio = ratio(parseInt(Height), parseInt(Width));

        noMedia.style.visibility = "hidden";
        if (sessionStorage.getItem('currentTab') == 'restoration') {
            mediaContainerRes.style.visibility = "visible";
        }

        mediaHeaderRes.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

        console.log(data);

        infoFormatRes.textContent = container + ' - ' + Format;
        infoSizeRes.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
        infoDimensionsRes.textContent = Width + " x " + Height + " (" + aspectRatio + ")";
        infoFrameRateRes.textContent = FrameRate;
        infoDurationRes.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
        infoFramecountRes.textContent = FrameCount;
        infoBitrateRes.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
        webFrame.clearCache()
        ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbRestoration.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
        var thumbRestoration = document.getElementById('restoration-thumb-img');
        function setThumbRestoration() {
            thumbRestoration.src = path.join(appDataPath, '/.enhancr/thumbs/thumbRestoration.png?' + Math.random() * 1000);
        }
        setTimeout(setThumbRestoration, 1000)
    })
};

const preview = document.getElementById('preview');
const duration = document.getElementById('duration');
const durationUp = document.getElementById('durationUp');
const durationRes = document.getElementById('durationRes');
const timecodeCurrent = document.getElementById('timecode-current')
const previewCurrentFrame = document.getElementById('framecount-current');
const frameCountUp = document.getElementById('frame-countUp');
const frameCountRes = document.getElementById('frame-countRes');

// update preview frames
var previewUpdater = setInterval(function () {
    var status = sessionStorage.getItem('status');
    var previewInitialized = sessionStorage.getItem('previewInitialized');
    if (previewInitialized == "true") {
        if (status == "interpolating") {
            timecodeDuration.innerHTML = duration.innerHTML;
            timecodeCurrent.innerHTML = new Date(preview.currentTime * 1000).toISOString().substr(14, 5);
            previewCurrentFrame.innerHTML = sessionStorage.getItem('currentFrame');
            previewTotalFrames.innerHTML = sessionStorage.getItem('totalFrames');
        }
        if (status == "upscaling") {
            timecodeDuration.innerHTML = durationUp.innerHTML;
            timecodeCurrent.innerHTML = new Date(preview.currentTime * 1000).toISOString().substr(14, 5);
            previewCurrentFrame.innerHTML = sessionStorage.getItem('currentFrame');
            previewTotalFrames.innerHTML = frameCountUp.innerHTML;
        }
        if (status == "restoring") {
            timecodeDuration.innerHTML = durationRes.innerHTML;
            timecodeCurrent.innerHTML = new Date(preview.currentTime * 1000).toISOString().substr(14, 5);
            previewCurrentFrame.innerHTML = sessionStorage.getItem('currentFrame');
            previewTotalFrames.innerHTML = frameCountRes.innerHTML;
        }
    }
}, 1000);

class Media {
    // fetch metadata (interpolation)
    static fetchMetadata() {
        let inputPath = sessionStorage.getItem('currentFile');
        readMetadata(inputPath).then((result) => {
            const data = JSON.parse(result);

            const video = data.media.track.find((item) => item["@type"] === "Video");
            const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

            const container = data.media.track[0].Format;
            const filesize = data.media.track[0].FileSize;
            const bitRate = data.media.track[0].OverallBitRate;

            const aspectRatio = ratio(parseInt(Height), parseInt(Width));

            mediaContainer.style.visibility = "visible";
            mediaContainerUp.style.visibility = "hidden";
            mediaContainerRes.style.visibility = "hidden";

            mediaHeader.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

            console.log(data);

            infoFormat.textContent = container + ' - ' + Format;
            infoSize.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
            infoDimensions.textContent = Width + " x " + Height + " (" + aspectRatio + ")";
            infoFrameRate.textContent = FrameRate + " ➜ " + parseFloat(FrameRate) * 2;
            infoDuration.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
            infoFramecount.textContent = FrameCount;
            infoBitrate.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
            webFrame.clearCache()
            ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbInterpolation.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
            var thumbInterpolation = document.getElementById('interpolation-thumb-img');
            function setThumbInterpolation() {
                thumbInterpolation.src = path.join(appDataPath, '/.enhancr/thumbs/thumbInterpolation.png?' + Math.random() * 1000);
            }
            setTimeout(setThumbInterpolation, 1000);
        })
    }
    // fetch metadata (upscaling)
    static fetchMetadataUpscale() {
        let inputPath = sessionStorage.getItem('currentFile');
        let upscaleFactor = parseInt(sessionStorage.getItem("currentFactor"));
        readMetadata(inputPath).then((result) => {
            const data = JSON.parse(result);

            const video = data.media.track.find((item) => item["@type"] === "Video");
            const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

            const container = data.media.track[0].Format;
            const filesize = data.media.track[0].FileSize;
            const bitRate = data.media.track[0].OverallBitRate;

            const aspectRatio = ratio(parseInt(Height), parseInt(Width));

            noMedia.style.visibility = "hidden";

            mediaContainer.style.visibility = "hidden";
            mediaContainerUp.style.visibility = "visible";
            mediaContainerRes.style.visibility = "hidden";

            mediaHeaderUp.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

            console.log(data);

            infoFormatUp.textContent = container + ' - ' + Format;
            infoSizeUp.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
            infoDimensionsUp.textContent = Width + " x " + Height + " ➜ " + parseInt(Width) * upscaleFactor + " x " + parseInt(Height) * upscaleFactor + " (" + aspectRatio + ")";
            infoFrameRateUp.textContent = FrameRate;
            infoDurationUp.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
            infoFramecountUp.textContent = FrameCount;
            infoBitrateUp.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
            webFrame.clearCache()
            ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbUpscaling.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
            var thumbUpscaling = document.getElementById('upscaling-thumb-img');
            function setThumbUpscaling() {
                thumbUpscaling.src = path.join(appDataPath, '/.enhancr/thumbs/thumbUpscaling.png?' + Math.random() * 1000);
            }
            setTimeout(setThumbUpscaling, 1000)
        })
    }
    static fetchMetadataRestore() {
        let inputPath = sessionStorage.getItem('currentFile');
        readMetadata(inputPath).then((result) => {
            const data = JSON.parse(result);

            const video = data.media.track.find((item) => item["@type"] === "Video");
            const { Format, FrameRate, FrameCount, Width, Height, Duration } = video;

            const container = data.media.track[0].Format;
            const filesize = data.media.track[0].FileSize;
            const bitRate = data.media.track[0].OverallBitRate;

            const aspectRatio = ratio(parseInt(Height), parseInt(Width));

            mediaContainer.style.visibility = "hidden";
            mediaContainerUp.style.visibility = "hidden";
            mediaContainerRes.style.visibility = "visible";

            mediaHeaderRes.innerHTML = '<i class="fa-solid fa-folder-closed"></i> ../' + path.basename(inputPath);

            console.log(data);

            infoFormatRes.textContent = container + ' - ' + Format;
            infoSizeRes.textContent = Math.round(parseInt(filesize) / (1024 * 1024) * 100) / 100 + ' MB';
            infoDimensionsRes.textContent = Width + " x " + Height + " (" + aspectRatio + ")";
            infoFrameRateRes.textContent = FrameRate;
            infoDurationRes.textContent = new Date(parseInt(Duration) * 1000).toISOString().substr(14, 5);
            infoFramecountRes.textContent = FrameCount;
            infoBitrateRes.textContent = Math.round(parseInt(bitRate) / 1024) + " kbit/s"
            webFrame.clearCache()
            ffmpeg(inputPath).screenshots({ count: 1, filename: 'thumbRestoration.png', folder: path.join(appDataPath, '/.enhancr/thumbs'), size: '1280x720' });
            var thumbRestoration = document.getElementById('restoration-thumb-img');
            function setThumbRestoration() {
                thumbRestoration.src = path.join(appDataPath, '/.enhancr/thumbs/thumbRestoration.png?' + Math.random() * 1000);
            }
            setTimeout(setThumbRestoration, 1000)
        })
    }
}

module.exports = Media;