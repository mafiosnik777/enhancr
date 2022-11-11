const fs = require('fs-extra');
const os = require('os');
const path = require('path');

var preview = document.getElementById('preview');

function getTmpPath() {
    if (process.platform == 'win32') {
        return os.tmpdir() + "\\enhancr\\";
    } else {
        return os.tmpdir() + "/enhancr/";
    }
}

var tempPath = getTmpPath();
var previewPath = path.join(tempPath, '/preview');

class Preview {
    static listenForPreview() {
        var previewInterval = setInterval(function () {
            if (document.getElementById('preview-check').checked) {
                if (fs.existsSync(path.join(previewPath, '/master.m3u8'))) {
                    if (Hls.isSupported()) {
                        var hls = new Hls();
                        hls.loadSource(path.join(previewPath, '/master.m3u8'));
                        hls.attachMedia(preview);
                        hls.on(Hls.Events.MANIFEST_PARSED, function () {
                            preview.play();
                        });
                    }
                    sessionStorage.setItem('previewInitialized', 'true');
                    clearInterval(previewInterval);
                } else {
                    console.log("Preview not initialized yet.")
                }
            } else {
                // do nothing
            }
        }, 1000);
    }
}

module.exports = Preview;
