var fse = require('fs-extra');
var path = require('path');
const os = require('os');

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

const terminal = document.getElementById("terminal-text");

const diff = (diffBefore, diffAfter) => diffBefore.split(diffAfter).join('')

window.addEventListener("error", (event) => {
    fse.appendFileSync(path.join(appDataPath, '/.enhancr/', 'log.txt'), '\n\n [renderer process error] \n\n' + stringifyObject(event) + '\n\n');
});

fse.writeFile(path.join(appDataPath, '/.enhancr/', 'log.txt'), 'enhancr build 0.9.3a (pre-release)\n\n' + terminal.innerHTML);
fse.writeFile(os.tmpdir() + '/tmpLog.txt', terminal.innerHTML);

async function log() {
    let tmpLog = fse.readFileSync(os.tmpdir() + '/tmpLog.txt', {
        encoding: "utf8"
    });
    fse.writeFile(os.tmpdir() + '/tmpLog.txt', terminal.innerHTML);
    let terminalLog = diff(terminal.innerHTML, tmpLog);
    fse.appendFile(path.join(appDataPath, '/.enhancr/', 'log.txt'), terminalLog);
}
setInterval(log, 5000);

function stringifyObject(object, depth = 0, max_depth = 2) {
    // change max_depth to see more levels, for a touch event, 2 is good
    if (depth > max_depth)
        return 'Object';

    const obj = {};
    for (let key in object) {
        let value = object[key];
        if (value instanceof Node)
            // specify which properties you want to see from the node
            value = {
                id: value.id
            };
        else if (value instanceof Window)
            value = 'Window';
        else if (value instanceof Object)
            value = stringifyObject(value, depth + 1, max_depth);

        obj[key] = value;
    }

    return depth ? obj : JSON.stringify(obj);
}