const os = require('os');
const isWin11 = os.release().split('.')[2] >= 22000;
module.exports = (useAcryllic) => (
    process.platform === 'win32' && useAcryllic || process.platform === 'win32' && isWin11
        ? require('electron-acrylic-window').BrowserWindow
        : require('electron').BrowserWindow
);
