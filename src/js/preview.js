const fs = require('fs-extra');
const os = require('os');
const path = require('path');

const terminal = document.getElementById("terminal-text");
var preview = document.getElementById('preview');

class Preview {
    static listenForPreview() {
        var previewInterval = setInterval(function () {
            if (document.getElementById('preview-check').checked) {
                let cacheInputText = document.getElementById('cache-input-text');
                var cache = path.normalize(cacheInputText.textContent);
                var previewPath = path.join(cache, '/preview');
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
