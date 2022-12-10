module.exports = (useAcryllic) => (
    process.platform === 'win32' && useAcryllic
        ? require('electron-acrylic-window').BrowserWindow
        : require('electron').BrowserWindow
);
