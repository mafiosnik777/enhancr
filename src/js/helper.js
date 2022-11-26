const os = require('os');
const fse = require('fs-extra');
const path = require("path");
const { ipcRenderer } = require("electron");

const find = require('find-process');

const execSync = require('child_process').execSync;
const exec = require('child_process').exec;
const { spawn } = require('child_process');

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
let tempPath = getTmpPath();

let previewPath = path.join(tempPath, '/preview');
let previewDataPath = previewPath + '/data%02d.ts';

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

class enhancr {

    static version = '0.9.0 (pre-release)'

    static terminal(string, prefix = true, newLine = true, error = false) {
        let prfx = prefix ? "[enhancr] " : "";
        let nl = newLine ? "\n" : "";
        let err = error ? "[Error] " : "";
        let str = error ? `${nl}${err}${string}` : `${nl}${prfx}${string}`
        terminal.innerHTML += str;
    }

    static interpolate(item) {
        
    }

    static upscale(item) {

    }

    static restore(item) {

    }
}

module.exports = enhancr
