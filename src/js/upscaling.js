const fse = require('fs-extra');
const os = require('os');
const path = require("path");
const { ipcRenderer } = require("electron");

const find = require('find-process');

const execSync = require('child_process').execSync;
const exec = require('child_process').exec;
const { spawn } = require('child_process');

const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("@ffmpeg-installer/ffmpeg").path;
const ffprobePath = require("@ffprobe-installer/ffprobe").path;
ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);

const terminal = document.getElementById("terminal-text");
const enhancrPrefix = "[enhancr]";
const progressSpan = document.getElementById("progress-span");

function getTmpPath() {
    if (process.platform == 'win32') {
        return os.tmpdir() + "\\enhancr\\";
    } else {
        return os.tmpdir() + "/enhancr/";
    }
}
let temp = getTmpPath();
let previewPath = path.join(temp, '/preview');
let previewDataPath = previewPath + '/data%02d.ts';
const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

const blankModal = document.querySelector("#blank-modal");
const modal = document.querySelector("#modal");
const processOverlay = document.getElementById("process-overlay");

function openModal(modal) {
    if (modal == undefined) return
    modal.classList.add('active')
    overlay.classList.add('active')
}

const successModal = document.getElementById("modal-success");
const successTitle = document.getElementById("success-title");
const thumbModal = document.getElementById("thumb-modal");

const preview = document.getElementById('preview-check');

sessionStorage.setItem('stopped', 'false');

class Upscaling {
    static async process(scale, dimensions, file, output, params, extension, engine, fileOut, index) {
        let stopped = sessionStorage.getItem('stopped');
        if (!(stopped == 'true')) {
            // set flag for started upscaling process
            sessionStorage.setItem('status', 'upscaling');
            sessionStorage.setItem('engine', engine);

            // render progresbar
            const loading = document.getElementById("loading");
            loading.style.visibility = "visible";

            // check if output path field is filled
            if (document.getElementById('upscale-output-path-text').innerHTML == '') {
                openModal(blankModal);
                terminal.innerHTML += "\r\n[Error] Output path not specified, cancelling.";
                sessionStorage.setItem('status', 'error');
                processOverlay.style.visibility = "hidden";
            }

            // create paths if not existing
            if (!fse.existsSync(temp)) {
                fse.mkdirSync(temp);
            };

            if (!fse.existsSync(output)) {
                fse.mkdirSync(output)
            };

            // clear temporary files
            fse.emptyDirSync(temp);
            console.log(enhancrPrefix + " tmp directory cleared");

            if (!fse.existsSync(previewPath)) {
                fse.mkdirSync(previewPath);
            };

            terminal.innerHTML += '\r\n' + enhancrPrefix + ' Preparing media for upscaling process..';

            // scan media for subtitles
            const subsPath = path.join(temp, "subs.ass");
            try {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scanning media for subtitles..`;
                execSync(`ffmpeg -y -loglevel error -i ${file} -map 0:s:0 ${subsPath}`);
            } catch (err) {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` No subtitles were found, skipping subtitle extraction..`;
            };

            // scan media for audio
            const audioPath = path.join(temp, "audio.mka");
            try {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scanning media for audio..`;
                execSync(`ffmpeg -y -loglevel quiet -i "${file}" -vn -c copy ${audioPath}`)
            } catch (err) {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` No audio stream was found, skipping copying audio..`;
            };

            //get trtexec path
            function getTrtExecPath() {
                return path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/vsmlrt-cuda/trtexec.exe");
            }
            let trtexec = getTrtExecPath();

            let fp16 = document.getElementById('fp16-check');

            //get onnx input path
            function getOnnxPath() {
                if (!(document.getElementById('custom-model-check').checked)) {
                    if (engine == 'Upscaling - RealESRGAN (TensorRT)') {
                        return path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/esrgan/animevideov3.onnx");
                    } else if (engine == 'Upscaling - AnimeSR (TensorRT)') {
                        return path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/animesr/animesr_v2.onnx");
                    }

                } else {
                    terminal.innerHTML += '\r\n[enhancr] Using custom model: ' + path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML);
                    return path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML);
                }
            }
            let onnx = getOnnxPath();

            let floatingPoint = document.getElementById('fp16-check').checked;
            let fp = floatingPoint ? "fp16" : "fp32";

            let shapeOverride = document.getElementById('shape-check').checked;
            let shapeDimensionsMax = shapeOverride ? document.getElementById('shape-res').value : '1080x1920';
            let shapeDimensionsOpt = Math.ceil(parseInt(shapeDimensionsMax.split('x')[0]) / 2) + 'x' + Math.ceil(parseInt(shapeDimensionsMax.split('x')[1]) / 2);

            // get engine path
            function getEnginePath() {
                return path.join(appDataPath, '/.enhancr/models/engine', path.parse(onnx).name + '-' + fp + '_' + shapeDimensionsMax + '.engine');
            }
            let engineOut = getEnginePath();
            sessionStorage.setItem('engineOut', engineOut);

            // convert onnx to trt engine
            if (!fse.existsSync(engineOut) && engine == 'Upscaling - RealESRGAN (TensorRT)' || !fse.existsSync(engineOut) && engine == 'Upscaling - AnimeSR (TensorRT)') {
                function convertToEngine() {
                    return new Promise(function (resolve) {
                        if (engine == 'Upscaling - RealESRGAN (TensorRT)') {
                            if (fp16.checked == true) {
                                var cmd = `"${trtexec}" --fp16 --onnx="${onnx}" --minShapes=input:1x3x8x8 --optShapes=input:1x3x${shapeDimensionsOpt} --maxShapes=input:1x3x${shapeDimensionsMax} --saveEngine="${engineOut}" --verbose --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                            } else {
                                var cmd = `"${trtexec}" --onnx="${onnx}" --minShapes=input:1x3x8x8 --optShapes=input:1x3x${shapeDimensionsOpt} --maxShapes=input:1x3x${shapeDimensionsMax} --saveEngine="${engineOut}" --verbose --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                            }
                        } else {
                            var cmd = `"${trtexec}" --onnx="${onnx}" --optShapes=input:1x6x${shapeDimensionsOpt} --saveEngine="${engineOut}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                        }
                        let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'], windowsHide: true });
                        process.stdout.write('');
                        term.stdout.on('data', (data) => {
                            process.stdout.write(`${data}`);
                            terminal.innerHTML += data;
                        });
                        term.stderr.on('data', (data) => {
                            process.stderr.write(`${data}`);
                            progressSpan.innerHTML = path.basename(file) + ' | Converting onnx to engine..';
                            terminal.innerHTML += data;
                        });
                        term.on("close", () => {
                            sessionStorage.setItem('conversion', 'success');
                            resolve();
                        });
                    })
                }
                await convertToEngine();
            }

            // display infos in terminal for user
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Encoding parameters: ${params}`;
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scale: ${scale}x`;
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Mode: ${engine}`;

            const numStreams = document.getElementById('num-streams');

            const ffmpeg = path.join(__dirname, '..', "python/ffmpeg/ffmpeg.exe");

            // trim video if timestamps are set by user
            if (!(sessionStorage.getItem(`trim${index}`) == null)) {
                terminal.innerHTML += '\r\n[enhancr] Trimming video with timestamps ' + '"' + sessionStorage.getItem(`trim${index}`) + '"';
                let timestampStart = (sessionStorage.getItem(`trim${index}`)).split('-')[0];
                let timestampEnd = (sessionStorage.getItem(`trim${index}`)).split('-')[1];
                let trimmedOut = path.join(temp, path.parse(file).name + '.mkv');
                try {
                    function trim() {
                        return new Promise(function (resolve) {
                            if (document.getElementById("trim-check").checked) {
                                var cmd = `"${ffmpeg}" -y -loglevel error -ss ${timestampStart} -to ${timestampEnd} -i "${file}" -c copy -c:v libx264 -crf 14 -max_interleave_delta 0 "${trimmedOut}"`;
                            } else {
                                var cmd = `"${ffmpeg}" -y -loglevel error -ss ${timestampStart} -to ${timestampEnd} -i "${file}" -c copy -max_interleave_delta 0 "${trimmedOut}"`;
                            }
                            let term = spawn(cmd, [], {
                                shell: true,
                                stdio: ['inherit', 'pipe', 'pipe'],
                                windowsHide: true
                            });
                            process.stdout.write('');
                            term.stdout.on('data', (data) => {
                                process.stdout.write(`${data}`);
                                terminal.innerHTML += data;
                            });
                            term.stderr.on('data', (data) => {
                                process.stderr.write(`${data}`);
                                terminal.innerHTML += data;
                            });
                            term.on("close", () => {
                                file = trimmedOut;
                                terminal.innerHTML += '\r\n[enhancr] Trimmed video successfully.';
                                resolve();
                            })
                        })
                    }
                    await trim();
                } catch (error) {
                    terminal.innerHTML('[Trim] ' + error)
                }
            }

            // temp file for passing info to the AI
            const jsonPath = path.join(temp, "tmp.json");
            let json = {
                file: file,
                engine: engineOut,
                scale: scale,
                streams: numStreams.value,
                onnx: onnx,
                tiling: document.getElementById("tiling-check").checked,
                tileHeight: (document.getElementById("tile-res").value).split('x')[1],
                tileWidth: (document.getElementById("tile-res").value).split('x')[0]
            };
            let data = JSON.stringify(json);
            // write data to json
            fse.writeFileSync(jsonPath, data, (err) => {
                if (err) {
                    console.log("Error writing file", err);
                };
            });

            let model;
            // determine model
            if (engine == "Upscaling - RealESRGAN (TensorRT)") {
                model = "RealESRGAN"
            } else if (engine == "Upscaling - waifu2x (NCNN)") {
                model = "waifu2x"
            } else if (engine == "Upscaling - AnimeSR (TensorRT)") {
                model = "AnimeSR"
            }
            // resolve output file path
            if (fileOut == null) {
                let outPath = path.join(output, path.parse(file).name + `_${model}-${scale}x${extension}`);
                sessionStorage.setItem("pipeOutPath", outPath);
            } else {
                sessionStorage.setItem("pipeOutPath", `${path.join(output, fileOut + extension)}`);
            }

            // determine ai engine
            function pickEngine() {
                if (engine == "Upscaling - RealESRGAN (TensorRT)") {
                    return path.join(__dirname, '..', "/python/esrgan.py");
                }
                if (engine == "Upscaling - waifu2x (NCNN)") {
                    return path.join(__dirname, '..', "/python/waifu2x.py");
                }
                if (engine == "Upscaling - AnimeSR (TensorRT)") {
                    return path.join(__dirname, '..', "/python/animesr_trt.py");
                }
            }
            var engine = pickEngine();

            // determine vspipe path
            function pickVspipe() {
                if (process.platform == "win32") {
                    if (document.getElementById('python-check').checked) {
                        return "vspipe"
                    } else {
                        return path.join(__dirname, '..', "\\python\\bin\\VSPipe.exe")
                    }
                }
                if (process.platform == "linux") {
                    return "vspipe"
                }
                if (process.platform == "darwin") {
                    return "vspipe"
                }
            }
            let vspipe = pickVspipe();

            // get width & height
            function getWidth() {
                return parseInt((dimensions).split(' x')[0]) * scale;
            };
            let width = getWidth();

            function getHeight() {
                return parseInt(((dimensions).split('x ')[1]).split(' ')[0]) * scale;
            }
            let height = getHeight();

            let tmpOutPath = path.join(temp, Date.now() + extension);
            if (extension != ".mkv" && fse.existsSync(subsPath) == true) {
                openModal(modal);
                terminal.innerHTML += "\r\n[enhancr] Input video contains subtitles, but output container is not .mkv, cancelling.";
                sessionStorage.setItem('status', 'error');
                processOverlay.style.visibility = "hidden";
            } else {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Starting upscaling process..` + '\r\n';

                function upscale() {
                    return new Promise(function (resolve) {
                        // if preview is enabled split out 2 streams from output
                        if (preview.checked == true) {
                            var cmd = `"${vspipe}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} -s ${width}x${height} "${tmpOutPath}" -f hls -hls_list_size 0 -hls_flags independent_segments -hls_time 0.5 -hls_segment_type mpegts -hls_segment_filename "${previewDataPath}" -preset veryfast -vf scale=960:-1 "${path.join(previewPath, '/master.m3u8')}"`;
                        } else {
                            var cmd = `"${vspipe}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} -s ${width}x${height} "${tmpOutPath}"`;
                        }
                        let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'], windowsHide: true });
                        // merge stdout & stderr & write data to terminal
                        process.stdout.write('');
                        term.stdout.on('data', (data) => {
                            process.stdout.write(`[Pipe] ${data}`);
                        });
                        term.stderr.on('data', (data) => {
                            process.stderr.write(`[Pipe] ${data}`);
                            terminal.innerHTML += '[Pipe] ' + data;
                            sessionStorage.setItem('progress', data);
                        });
                        term.on("close", () => {
                            let lines = terminal.value.match(/[^\r\n]+/g);
                            let log = lines.slice(-10).reverse();
                            // don't merge streams if an error occurs
                            if (log.includes('[Pipe] pipe:: Invalid data found when processing input')) {
                                terminal.innerHTML += `[enhancr] An error has occured.`;
                                sessionStorage.setItem('status', 'done');
                                resolve();
                            } else {
                                terminal.innerHTML += `[enhancr] Finishing up upscaling..\r\n`;
                                terminal.innerHTML += `[enhancr] Muxing in streams..\r\n`;

                                // fix audio loss when muxing mkv
                                let mkv = extension == ".mkv";
                                let mkvFix = mkv ? "-max_interleave_delta 0" : "";

                                let muxCmd = `"${ffmpeg}" -y -loglevel error -i "${file}" -i ${tmpOutPath} -map 1 -map 0 -map -0:v -codec copy ${mkvFix} "${sessionStorage.getItem('pipeOutPath')}"`;
                                let muxTerm = spawn(muxCmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'], windowsHide: true });

                                // merge stdout & stderr & write data to terminal
                                process.stdout.write('');
                                muxTerm.stdout.on('data', (data) => {
                                    process.stdout.write(`[Pipe] ${data}`);
                                });
                                muxTerm.stderr.on('data', (data) => {
                                    process.stderr.write(`[Pipe] ${data}`);
                                    terminal.innerHTML += '[Pipe] ' + data;
                                    sessionStorage.setItem('progress', data);
                                });
                                muxTerm.on("close", () => {
                                    // finish up upscaling process
                                    terminal.innerHTML += `[enhancr] Completed upscaling`;
                                    const upscalingBtnSpan = document.getElementById("upscaling-button-text");
                                    var notification = new Notification("Upscaling completed", { icon: "./assets/enhancr.png", body: path.basename(file) });
                                    sessionStorage.setItem('status', 'done');
                                    successTitle.innerHTML = path.basename(sessionStorage.getItem("upscaleInputPath"));
                                    thumbModal.src = path.join(appDataPath, '/.enhancr/thumbs/thumbUpscaling.png?' + Date.now());
                                    resolve();
                                });
                            }
                        });
                    });
                }
                await upscale();
                // clear temporary files
                // fse.emptyDirSync(temp);
                console.log("Cleared temporary files");
                // timeout for 2 seconds after upscale
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
}

module.exports = Upscaling;
