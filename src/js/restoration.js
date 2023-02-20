const fse = require('fs-extra');
const os = require('os');
const path = require("path");
const { ipcRenderer } = require("electron");

const find = require('find-process');

const execSync = require('child_process').execSync;
const exec = require('child_process').exec;
const { spawn } = require('child_process');

const remote = require('@electron/remote');
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

const terminal = document.getElementById("terminal-text");
const enhancrPrefix = "[enhancr]";
const progressSpan = document.getElementById("progress-span");

const blankModal = document.getElementById("blank-modal");
const subsModal = document.getElementById("modal");
const processOverlay = document.getElementById("process-overlay");

function openModal(modal) {
    if (modal == undefined) return
    modal.classList.add('active')
    overlay.classList.add('active')
}

const isPackaged = remote.app.isPackaged;

const successModal = document.getElementById("modal-success");
const successTitle = document.getElementById("success-title");
const thumbModal = document.getElementById("thumb-modal");

const preview = document.getElementById('preview-check');

sessionStorage.setItem('stopped', 'false');

class Restoration {
    static async process(file, model, output, params, extension, engine, fileOut, index) {
        let cacheInputText = document.getElementById('cache-input-text');
        var cache = path.normalize(cacheInputText.textContent);

        let previewPath = path.join(cache, '/preview');
        let previewDataPath = previewPath + '/data%02d.ts';
        const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

        let stopped = sessionStorage.getItem('stopped');
        if (!(stopped == 'true')) {
            // set flag for started restoration process
            sessionStorage.setItem('status', 'restoring');
            sessionStorage.setItem('engine', engine);

            // render progresbar
            const loading = document.getElementById("loading");
            loading.style.visibility = "visible";

            // check if output path field is filled
            if (document.getElementById('restore-output-path-text').innerHTML == '') {
                openModal(blankModal);
                terminal.innerHTML += "\r\n[Error] Output path not specified, cancelling.";
                sessionStorage.setItem('status', 'error');
                throw new Error('Output path not specified');
            }

            // create paths if not existing
            if (!fse.existsSync(cache)) {
                fse.mkdirSync(cache);
            };

            if (!fse.existsSync(output)) {
                fse.mkdirSync(output)
            };

            // clear temporary files
            fse.emptyDirSync(cache);
            console.log(enhancrPrefix + " tmp directory cleared");

            if (!fse.existsSync(previewPath)) {
                fse.mkdirSync(previewPath);
            };

            terminal.innerHTML += '\r\n' + enhancrPrefix + ' Preparing media for restoration process..';

            const ffmpeg = !isPackaged ? path.join(__dirname, '..', "env/ffmpeg/ffmpeg.exe") : path.join(process.resourcesPath, "env/ffmpeg/ffmpeg.exe");

            // scan media for subtitles
            const subsPath = path.join(cache, "subs.ass");
            try {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scanning media for subtitles..`;
                execSync(`${ffmpeg} -y -loglevel error -i "${file}" -c:s copy ${subsPath}`);
            } catch (err) {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` No subtitles were found, skipping subtitle extraction..`;
            };

            // scan media for audio
            const audioPath = path.join(cache, "audio.mka");
            try {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scanning media for audio..`;
                execSync(`${ffmpeg} -y -loglevel quiet -i "${file}" -vn -c copy ${audioPath}`)
            } catch (err) {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` No audio stream was found, skipping copying audio..`;
            };

            //get trtexec path
            function getTrtExecPath() {
                return !isPackaged ? path.join(__dirname, '..', "/env/python/Library/bin/trtexec.exe") : path.join(process.resourcesPath, "/env/python/Library/bin/trtexec.exe")
            }
            let trtexec = getTrtExecPath();

            //get python path
            function getPythonPath() {
                return !isPackaged ? path.join(__dirname, '..', "/env/python/python.exe") : path.join(process.resourcesPath, "/env/python/python.exe");
            }
            let python = getPythonPath();

            //get conversion script
            function getConversionScript() {
                return !isPackaged ? path.join(__dirname, '..', "/env/utils/convert_model_esrgan.py") : path.join(process.resourcesPath, "/env/utils/convert_model_esrgan.py")
            }
            let convertModel = getConversionScript();

            var customModel = path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML);

            // convert pth to onnx
            if (document.getElementById('custom-model-check').checked && path.extname(customModel) == ".pth") {
                function convertToOnnx() {
                    return new Promise(function (resolve) {
                        var cmd = `"${python}" "${convertModel}" --input="${path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML)}" --output="${path.join(cache, path.parse(customModel).name + '.onnx')}"`;
                        let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'], windowsHide: true });
                        process.stdout.write('');
                        term.stdout.on('data', (data) => {
                            process.stdout.write(`${data}`);
                            terminal.innerHTML += data;
                        });
                        term.stderr.on('data', (data) => {
                            process.stderr.write(`${data}`);
                            progressSpan.innerHTML = path.basename(file) + ' | Converting pth to onnx..';
                            terminal.innerHTML += data;
                        });
                        term.on("close", () => {
                            resolve();
                        });
                    })
                }
                await convertToOnnx();
            }

            //get onnx input path
            function getOnnxPath() {
                if (engine == 'Restoration - DPIR (TensorRT)' && model == 'Denoise') {
                    return !isPackaged ? path.join(__dirname, '..', "/env/python/vapoursynth64/plugins/models/dpir/dpir_denoise.onnx") : path.join(process.resourcesPath, "/env/python/vapoursynth64/plugins/models/dpir/dpir_denoise.onnx")
                } else if (engine == 'Restoration - DPIR (TensorRT)' && model == 'Deblock') {
                    return !isPackaged ? path.join(__dirname, '..', "/env/python/vapoursynth64/plugins/models/dpir/dpir_deblock.onnx") : path.join(process.resourcesPath, "/env/python/vapoursynth64/plugins/models/dpir/dpir_deblock.onnx")
                } else if (engine == 'Restoration - RealESRGAN (1x) (TensorRT)' && !(document.getElementById('custom-model-check').checked)) {
                    return !isPackaged ? path.join(__dirname, '..', "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx") : path.join(process.resourcesPath, "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx")
                } else if (engine == 'Restoration - RealESRGAN (1x) (NCNN)' && !(document.getElementById('custom-model-check').checked)) {
                    return !isPackaged ? path.join(__dirname, '..', "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx") : path.join(process.resourcesPath, "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx")
                } else {
                    terminal.innerHTML += '\r\n[enhancr] Using custom model: ' + customModel;
                    if (path.extname(customModel) == ".pth") {
                        return path.join(cache, path.parse(customModel).name + '.onnx');
                    } else {
                        return path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML);
                    }
                }
            }
            var onnx = getOnnxPath();

            let floatingPoint = document.getElementById('fp16-check').checked;
            let fp = floatingPoint ? "fp16" : "fp32";

            let shapeOverride = document.getElementById('shape-check').checked;
            let shapeDimensionsMax = shapeOverride ? document.getElementById('shape-res').value : '1080x1920';
            let shapeDimensionsOpt = Math.ceil(parseInt(shapeDimensionsMax.split('x')[0]) / 1.5) + 'x' + Math.ceil(parseInt(shapeDimensionsMax.split('x')[1]) / 1.5);

            // get engine path
            function getEnginePath() {
                if (engine == 'Restoration - RealESRGAN (1x) (NCNN)') {
                    if (!(document.getElementById('custom-model-check').checked)) {
                        return !isPackaged ? path.join(__dirname, '..', "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx") : path.join(process.resourcesPath, "/env/python/vapoursynth64/plugins/models/esrgan/animevideov3.onnx")
                    } else {
                        return path.join(appDataPath, '/.enhancr/models/RealESRGAN', document.getElementById('custom-model-text').innerHTML);
                    }
                } else {
                    return path.join(appDataPath, '/.enhancr/models/engine', path.parse(onnx).name + '-' + fp + '_' + shapeDimensionsMax + '_trt_8.5.2.engine');
                }
            }
            let engineOut = getEnginePath();
            sessionStorage.setItem('engineOut', engineOut);

            let fp16 = document.getElementById('fp16-check');

            let dim = () => {
                if (engine == 'Restoration - RealESRGAN (1x) (TensorRT)') return "3";
                else return "4";
            }

            // convert onnx to trt engine
            if (!fse.existsSync(engineOut) && engine != 'Restoration - RealESRGAN (1x) (NCNN)') {
                function convertToEngine() {
                    return new Promise(function (resolve) {
                        if (fp16.checked == true) {
                            var cmd = `"${trtexec}" --fp16 --onnx="${onnx}" --minShapes=input:1x${dim()}x8x8 --optShapes=input:1x${dim()}x${shapeDimensionsOpt} --maxShapes=input:1x${dim()}x${shapeDimensionsMax} --saveEngine="${engineOut}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --buildOnly --preview=+fasterDynamicShapes0805`;
                        } else {
                            var cmd = `"${trtexec}" --onnx="${onnx}" --minShapes=input:1x${dim()}x8x8 --optShapes=input:1x${dim()}x${shapeDimensionsOpt} --maxShapes=input:1x${dim()}x${shapeDimensionsMax} --saveEngine="${engineOut}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --buildOnly --preview=+fasterDynamicShapes0805`;
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
                    });
                }
                await convertToEngine();
            }

            // display infos in terminal for user
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Encoding parameters: ${params}`;
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Mode: ${engine}`;

            // resolve media framerate and pass to vsynth for working around VFR content
            let fps = parseFloat((document.getElementById('framerate').innerHTML).split(" ")[0]);

            const numStreams = document.getElementById('num-streams');

            // trim video if timestamps are set by user
            if (!(sessionStorage.getItem(`trim${index}`) == null)) {
                terminal.innerHTML += '\r\n[enhancr] Trimming video with timestamps ' + '"' + sessionStorage.getItem(`trim${index}`) + '"';
                let timestampStart = (sessionStorage.getItem(`trim${index}`)).split('-')[0];
                let timestampEnd = (sessionStorage.getItem(`trim${index}`)).split('-')[1];
                let trimmedOut = path.join(cache, path.parse(file).name + '.mkv');
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

            let modelCheck = model == 'Denoise';
            const denoiseStrength = document.getElementById('denoise-strength');
            const deblockStrength = document.getElementById('deblock-strength');

            let strengthParam = modelCheck ? denoiseStrength.value : deblockStrength.value;

            // cache file for passing info to the AI
            const jsonPath = path.join(cache, "tmp.json");
            let json = {
                file: file,
                engine: engineOut,
                framerate: fps,
                streams: numStreams.value,
                model: model,
                fp16: fp16.checked,
                strength: parseInt(strengthParam),
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

            // determine model
            if (engine == "Restoration - DPIR (TensorRT)") {
                model = "DPIR"
            } else if (engine == "Restoration - RealESRGAN (1x) (TensorRT)") {
                model = "RealESRGAN-1x"
            } else {
                model = "RealESRGAN-1x"
            }

            // resolve output file path
            if (fileOut == null) {
                let outPath = path.join(output, path.parse(file).name + `_${model}-1x${extension}`);
                sessionStorage.setItem("pipeOutPath", outPath);
            } else {
                sessionStorage.setItem("pipeOutPath", `${path.join(output, fileOut + extension)}`);
            }

            // determine ai engine
            function pickEngine() {
                if (engine == "Restoration - DPIR (TensorRT)") {
                    return !isPackaged ? path.join(__dirname, '..', "/env/inference/dpir.py") : path.join(process.resourcesPath, "/env/inference/dpir.py")
                }
                if (engine == "Restoration - RealESRGAN (1x) (TensorRT)") {
                    return !isPackaged ? path.join(__dirname, '..', "env/inference/esrgan.py") : path.join(process.resourcesPath, "/env/inference/esrgan.py")
                }
                if (engine == "Restoration - RealESRGAN (1x) (NCNN)") {
                    return !isPackaged ? path.join(__dirname, '..', "env/inference/esrgan_ncnn.py") : path.join(process.resourcesPath, "/env/inference/esrgan_ncnn.py")
                }
            }
            var engine = pickEngine();

            // determine vspipe path
            function pickVspipe() {
                if (process.platform == "win32") {
                    if (document.getElementById('python-check').checked) {
                        return "vspipe"
                    } else {
                        return !isPackaged ? path.join(__dirname, '..', "\\env\\python\\VSPipe.exe") : path.join(process.resourcesPath, "\\env\\python\\VSPipe.exe");
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

            let dimensions = document.getElementById('dimensionsRes');

            // get width & height
            function getWidth() {
                return parseInt((dimensions.innerHTML).split(' x')[0]);
            };
            let width = getWidth();

            function getHeight() {
                return parseInt(((dimensions.innerHTML).split('x ')[1]).split(' ')[0]);
            }
            let height = getHeight();

            // inject env hook
            let inject_env = !isPackaged ? `"${path.join(__dirname, '..', "\\env\\python\\condabin\\conda_hook.bat")}" && "${path.join(__dirname, '..', "\\env\\python\\condabin\\conda_auto_activate.bat")}"` : `"${path.join(process.resourcesPath, "\\env\\python\\condabin\\conda_hook.bat")}" && "${path.join(process.resourcesPath, "\\env\\python\\condabin\\conda_auto_activate.bat")}"`;

            let tmpOutPath = path.join(cache, Date.now() + extension);
            if (extension != ".mkv" && fse.existsSync(subsPath) == true) {
                openModal(subsModal);
                terminal.innerHTML += "\r\n[enhancr] Input video contains subtitles, but output container is not .mkv, cancelling.";
                sessionStorage.setItem('status', 'error');
                throw new Error('Input video contains subtitles, but output container is not .mkv');
            } else {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Starting restoration process..` + '\r\n';

                function restore() {
                    return new Promise(function (resolve) {
                        // if preview is enabled split out 2 streams from output
                        if (preview.checked == true) {
                            var cmd = `${inject_env} && "${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} -s ${width}x${height} "${tmpOutPath}" -f hls -hls_list_size 0 -hls_flags independent_segments -hls_time 0.5 -hls_segment_type mpegts -hls_segment_filename "${previewDataPath}" -preset veryfast -vf scale=960:-1 "${path.join(previewPath, '/master.m3u8')}"`;
                        } else {
                            var cmd = `${inject_env} && "${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} -s ${width}x${height} "${tmpOutPath}"`;
                        }
                        let term = spawn(cmd, [], { shell: true, stdio: ['inherit', 'pipe', 'pipe'], windowsHide: true });
                        // merge stdout & stderr & write data to terminal
                        process.stdout.write('');
                        term.stdout.on('data', (data) => {
                            process.stdout.write(`[Pipe] ${data}`);
                        });
                        term.stderr.on('data', (data) => {
                            process.stderr.write(`[Pipe] ${data}`);
                            // remove leading and trailing whitespace, including newline characters
                            let dataString = data.toString().trim();
                            if (dataString.startsWith('Frame:')) {
                                // Replace the last line of the textarea with the updated line
                                terminal.innerHTML = terminal.innerHTML.replace(/([\s\S]*\n)[\s\S]*$/, '$1' + '[Pipe] ' + dataString);
                            } else if (!(dataString.startsWith('CUDA lazy loading is not enabled.'))) {
                                terminal.innerHTML += '\n[Pipe] ' + dataString;
                            }
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
                                terminal.innerHTML += `[enhancr] Finishing up restoration..\r\n`;
                                terminal.innerHTML += `[enhancr] Muxing in streams..\r\n`;

                                // fix audio loss when muxing mkv
                                let mkv = extension == ".mkv";
                                let mkvFix = mkv ? "-max_interleave_delta 0" : "";

                                let muxCmd = `"${ffmpeg}" -y -loglevel error -i "${file}" -i "${tmpOutPath}" -map 1 -map 0 -map -0:v -dn -codec copy ${mkvFix} "${sessionStorage.getItem('pipeOutPath')}"`;
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
                                    // finish up restoration process
                                    terminal.innerHTML += `[enhancr] Completed restoring`;
                                    var notification = new Notification("Restoration completed", { icon: "./assets/enhancr.png", body: path.basename(file) });
                                    sessionStorage.setItem('status', 'done');
                                    ipcRenderer.send('rpc-done');
                                    successTitle.innerHTML = path.basename(sessionStorage.getItem("inputPathRestore"));
                                    thumbModal.src = path.join(appDataPath, '/.enhancr/thumbs/thumbRestoration.png?' + Date.now());
                                    resolve();
                                });
                            }
                        });
                    });
                }
                await restore();
                // clear temporary files
                fse.emptyDirSync(cache);
                console.log("Cleared temporary cache files");
                // timeout for 2 seconds after restoration
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
}

module.exports = Restoration;