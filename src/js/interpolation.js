const fse = require('fs-extra');
const os = require('os');
const path = require("path");
const { ipcRenderer } = require("electron");

const find = require('find-process');

const execSync = require('child_process').execSync;
const { spawn } = require('child_process');

const terminal = document.getElementById("terminal-text");
const enhancrPrefix = "[enhancr]";

const modal = document.querySelector("#modal");
const blankModal = document.querySelector("#blank-modal");
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

class Interpolation {
    static async process(file, model, output, params, extension, engine, fileOut, index) {
        let cacheInputText = document.getElementById('cache-input-text');
        var cache = path.normalize(cacheInputText.textContent);

        let previewPath = path.join(cache, '/preview');
        let previewDataPath = previewPath + '/data%02d.ts';
        const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

        let stopped = sessionStorage.getItem('stopped');
        if (!(stopped == 'true')) {
            // set flag for started interpolation process
            sessionStorage.setItem('status', 'interpolating');
            sessionStorage.setItem('engine', engine);

            // render progressbar
            const loading = document.getElementById("loading");
            loading.style.visibility = "visible";

            // check if output path field is filled
            if (document.getElementById('output-path-text').innerHTML == '') {
                openModal(blankModal);
                terminal.innerHTML += "\r\n[Error] Output path not specified, cancelling.";
                sessionStorage.setItem('status', 'error');
                processOverlay.style.visibility = "hidden";
            }

            // create paths if not existing
            if (!fse.existsSync(cache)) {
                fse.mkdirSync(cache);
            };
            if (!fse.existsSync(output)) {
                fse.mkdirSync(output)
            };

            // clear cacheorary files
            fse.emptyDirSync(cache);
            console.log("Cleared cacheorary files");

            if (!fse.existsSync(previewPath)) {
                fse.mkdirSync(previewPath);
            };

            terminal.innerHTML += '\r\n' + enhancrPrefix + ' Preparing media for interpolation process..';

            // scan media for subtitles
            const subsPath = path.join(cache, "subs.ass");
            try {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Scanning media for subtitles..`;
                execSync(`ffmpeg -y -loglevel error -i ${file} -c:s copy ${subsPath}`);
            } catch (err) {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` No subtitles were found, skipping subtitle extraction..`;
            };

            // scan media for audio
            const audioPath = path.join(cache, "audio.mka");
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

            // get width & height
            let dimensions = document.getElementById('dimensions');
            var width = parseInt((dimensions.innerHTML).split(' x')[0]);
            var height = parseInt(((dimensions.innerHTML).split('x ')[1]).split(' ')[0]);
            var padding = false;
            var roundedWidth;
            var roundedHeight;
            var toPadWidth = 0;
            var toPadHeight = 0;

            let floatingPoint = document.getElementById('fp16-check').checked;
            let fp = floatingPoint ? "fp16" : "fp32";

            let systemPython = document.getElementById('python-check').checked;
            

            // check if dimensions are divisible by 8
            function isDivisibleBy8(num) {
                return num % 8 === 0;
            }
            function roundUpToNextMultipleOf8(num) {
                return Math.ceil(num / 8) * 8;
            }
            
            if (!isDivisibleBy8(width) || !isDivisibleBy8(height)) padding = true;

            if (padding && engine == "Channel Attention - CAIN (TensorRT)") {
                roundedWidth = roundUpToNextMultipleOf8(width);
                roundedHeight = roundUpToNextMultipleOf8(height);
                toPadHeight = roundedHeight - height;
                toPadWidth = roundedWidth - width;
                width = roundedWidth;
                height = roundedHeight
            }

            if (systemPython == true) {
                var python = "python";
            }
            else {
                var python = path.join(__dirname, '..', "\\python\\bin\\python.exe");
            }

            var convert = path.join(__dirname, '..', "\\python/torch/convert.py");
            var cainModel = document.getElementById('model-span').innerHTML == 'RVP - v1.0';
            var onnx = cainModel ? path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/cain-rvpv1/rvpv1.onnx") : path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/cain-cvpv6/cvpv6.onnx");
            var groups = model ? 2 : 3;

            // get engine path
            function getEnginePath() {
                return path.join(appDataPath, '/.enhancr/models/engine', 'cain' + '_' + path.basename(onnx, '.onnx') + '_' + width + 'x' + height + '_' + fp + '.engine');
            }
            let engineOut = getEnginePath();
            sessionStorage.setItem('engineOut', engineOut);

            let fp16 = document.getElementById('fp16-check');

            const progressSpan = document.getElementById("progress-span");

            // engine conversion (onnx -> engine) (cain-trt)
            if (!fse.existsSync(engineOut) && engine == 'Channel Attention - CAIN (TensorRT)') {
                function convertToEngine() {
                    return new Promise(function (resolve) {
                            if (fp16.checked == true) {
                                var engineCmd = `"${trtexec}" --fp16 --onnx="${onnx}" --optShapes=input:1x6x${height}x${width} --saveEngine="${engineOut}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                            } else {
                                var engineCmd = `"${trtexec}" --onnx="${onnx}" --optShapes=input:1x6x${height}x${width} --saveEngine="${engineOut}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                            }
                            let engineTerm = spawn(engineCmd, [], {
                                shell: true,
                                stdio: ['inherit', 'pipe', 'pipe'],
                                windowsHide: true
                            });
                            process.stdout.write('');
                            engineTerm.stdout.on('data', (data) => {
                                process.stdout.write(`${data}`);
                                terminal.innerHTML += data;
                            });
                            engineTerm.stderr.on('data', (data) => {
                                process.stderr.write(`${data}`);
                                progressSpan.innerHTML = path.basename(file) + ' | Converting onnx to engine..';
                                terminal.innerHTML += data;
                            });
                            engineTerm.on("close", () => {
                                sessionStorage.setItem('conversion', 'success');
                                resolve();
                            });
                    })
                }
                await convertToEngine();
            }

            function getRifeOnnx() {
                if (document.getElementById('rife-tta-check').checked) {
                    return path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/rife-trt/rife46_ensembleTrue.onnx");
                } else {
                    return path.join(__dirname, '..', "/python/bin/vapoursynth64/plugins/models/rife-trt/rife46_ensembleFalse.onnx");
                }
            }
            let rifeOnnx = getRifeOnnx();

            let shapeOverride = document.getElementById('shape-check').checked;
            let shapeDimensionsMax = shapeOverride ? document.getElementById('shape-res').value : '1080x1920';
            let shapeDimensionsOpt = '720x1280';

            function getRifeEngine() {
                if (document.getElementById('rife-tta-check').checked) {
                    if (fp16.checked == true) {
                        return path.join(appDataPath, '/.enhancr/models/engine', `rife46_ensembleTrue_fp16_${shapeDimensionsMax}.engine`);
                    } else {
                        return path.join(appDataPath, '/.enhancr/models/engine', `rife46_ensembleTrue_${shapeDimensionsMax}.engine`);
                    }
                } else {
                    if (fp16.checked == true) {
                        return path.join(appDataPath, '/.enhancr/models/engine', `rife46_ensembleFalse_fp16_${shapeDimensionsMax}.engine`);
                    } else {
                        return path.join(appDataPath, '/.enhancr/models/engine', `rife46_ensembleFalse_${shapeDimensionsMax}.engine`);
                    }
                }
            }
            let rifeEngine = getRifeEngine();

            // engine conversion (onnx -> engine) (rife-trt)
            if (!fse.existsSync(rifeEngine) && engine == 'Optical Flow - RIFE (TensorRT)') {
                function convertToEngine() {
                    return new Promise(function (resolve) {
                        if (fp16.checked == true) {
                            var cmd = `"${trtexec}" --fp16 --onnx="${rifeOnnx}" --minShapes=input:1x8x8x8 --optShapes=input:1x8x${shapeDimensionsOpt} --maxShapes=input:1x8x${shapeDimensionsMax} --saveEngine="${rifeEngine}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
                        } else {
                            var cmd = `"${trtexec}" --onnx="${rifeOnnx}" --minShapes=input:1x8x8x8 --optShapes=input:1x8x720x1280${shapeDimensionsOpt} --maxShapes=input:1x8x${shapeDimensionsMax} --saveEngine="${rifeEngine}" --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT`;
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
                            progressSpan.innerHTML = path.basename(file) + ' | Converting onnx to engine..';
                            terminal.innerHTML += data;
                        });
                        term.on("close", () => {
                            sessionStorage.setItem('conversion', 'success');
                            resolve();
                        })
                    })
                }
                await convertToEngine();
            }

            // display infos in terminal for user
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Encoding parameters: ${params}`;
            terminal.innerHTML += '\r\n' + enhancrPrefix + ` Model: ${model}`;

            // resolve media framerate and pass to vsynth for working around VFR content
            let fps = parseFloat((document.getElementById('framerate').innerHTML).split(" ")[0]);

            const numStreams = document.getElementById('num-streams');

            const ffmpeg = path.join(__dirname, '..', "python/ffmpeg/ffmpeg.exe");

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

            // cache file for passing info to the AI
            const jsonPath = path.join(cache, "tmp.json");
            let json = {
                file: file,
                model: model,
                engine: engineOut,
                rife_engine: rifeEngine,
                framerate: fps,
                padding: padding,
                toPadWidth: toPadWidth,
                toPadHeight: toPadHeight,
                streams: numStreams.value,
                sensitivity: document.getElementById('sensitivity-check').checked,
                sensitivityValue: document.getElementById('sensitivity').value,
                rife_tta: document.getElementById("rife-tta-check").checked,
                rife_uhd: document.getElementById("rife-uhd-check").checked,
                sc: document.getElementById("sc-check").checked,
                skip: document.getElementById("skip-check").checked,
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
            if (engine == "Channel Attention - CAIN (NCNN)") {
                model = "CAIN"
            } else if (engine == "Optical Flow - RIFE (NCNN)") {
                model = "RIFE"
            } else if (engine == "Optical Flow - RIFE (TensorRT)") {
                model = "RIFE"
            } else if (engine == "Channel Attention - CAIN (TensorRT)") {
                model = "CAIN"
            }
            // resolve output file path
            if (fileOut == null) {
                let outPath = path.join(output, path.parse(file).name + `_${model}-2x${extension}`);
                sessionStorage.setItem("pipeOutPath", outPath);
            } else {
                sessionStorage.setItem("pipeOutPath", `${path.join(output, fileOut + extension)}`);
            }

            // determine ai engine
            function pickEngine() {
                if (engine == "Channel Attention - CAIN (TensorRT)") {
                    return path.join(__dirname, '..', "/python/cain_trt.py");
                }
                if (engine == "Channel Attention - CAIN (NCNN)") {
                    return path.join(__dirname, '..', "/python/cain.py");
                }
                if (engine == "Optical Flow - RIFE (NCNN)") {
                    return path.join(__dirname, '..', "/python/rife.py");
                }
                if (engine == "Optical Flow - RIFE (TensorRT)") {
                    return path.join(__dirname, '..', "/python/rife_trt.py");
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

            let tmpOutPath = path.join(cache, Date.now() + extension);
            if (extension != ".mkv" && fse.existsSync(subsPath) == true) {
                openModal(modal);
                terminal.innerHTML += "\r\n[Error] Input video contains subtitles, but output container is not .mkv, cancelling.";
                sessionStorage.setItem('status', 'error');
                processOverlay.style.visibility = "hidden";
            } else {
                terminal.innerHTML += '\r\n' + enhancrPrefix + ` Starting interpolation process..` + '\r\n';

                function interpolate() {
                    return new Promise(function (resolve) {
                        // if preview is enabled split out 2 streams from output
                        if (preview.checked == true) {
                            var cmd = `"${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} "${tmpOutPath}" -f hls -hls_list_size 0 -hls_flags independent_segments -hls_time 0.5 -hls_segment_type mpegts -hls_segment_filename "${previewDataPath}" -preset veryfast -vf scale=960:-1 "${path.join(previewPath, '/master.m3u8')}"`;
                        } else {
                            var cmd = `"${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} "${tmpOutPath}"`;
                        }
                        let term = spawn(cmd, [], {
                            shell: true,
                            stdio: ['inherit', 'pipe', 'pipe'],
                            windowsHide: true
                        });
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
                                terminal.innerHTML += `[enhancr] Finishing up interpolation..\r\n`;
                                terminal.innerHTML += `[enhancr] Muxing in streams..\r\n`;

                                // fix audio loss when muxing mkv
                                let mkv = extension == ".mkv";
                                let mkvFix = mkv ? "-max_interleave_delta 0" : "";

                                let out = sessionStorage.getItem('pipeOutPath');

                                let muxCmd = `"${ffmpeg}" -y -loglevel error -i "${file}" -i "${tmpOutPath}" -map 1 -map 0 -map -0:v -codec copy ${mkvFix} "${out}"`;
                                let muxTerm = spawn(muxCmd, [], {
                                    shell: true,
                                    stdio: ['inherit', 'pipe', 'pipe'],
                                    windowsHide: true
                                });

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
                                    // finish up interpolation process
                                    terminal.innerHTML += `[enhancr] Completed interpolation`;
                                    const interpolateBtnSpan = document.getElementById("interpolate-button-text");
                                    var notification = new Notification("Interpolation completed", {
                                        icon: "./assets/enhancr.png",
                                        body: path.basename(file)
                                    });
                                    sessionStorage.setItem('status', 'done');
                                    ipcRenderer.send('rpc-done');
                                    successTitle.innerHTML = path.basename(sessionStorage.getItem("inputPath"));
                                    thumbModal.src = path.join(appDataPath, '/.enhancr/thumbs/thumbInterpolation.png?' + Date.now());
                                    resolve();
                                });
                            }
                        })
                    })
                }
                await interpolate();
                // clear cacheorary files
                // fse.emptyDirSync(cache);
                console.log("Cleared temporary files");
                // timeout for 2 seconds after interpolation
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
}

module.exports = Interpolation;