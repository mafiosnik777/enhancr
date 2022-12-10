const { app, shell } = require('electron');

if (!app.requestSingleInstanceLock()) app.quit();

const remoteMain = require('@electron/remote/main');
const { setVibrancy } = require('electron-acrylic-window');

const fs = require('fs-extra');
const path = require('path');
const os = require('os');

const setupIpc = require('./main/setup-ipc');

const appDataPaths = {
    win32: process.env.APPDATA,
    darwin: `${process.env.HOME}/Library/Preferences`,
    linux: `${process.env.HOME}/.local/share`,
};

const createDirs = [
    'models/RealESRGAN',
    'models/engine',
    'thumbs',
];

const devMode = !app.isPackaged;
const initialHtml = devMode ? './pages/welcome.html' : './pages/launch.html';

const appDataPath = path.resolve(appDataPaths[process.platform], '.enhancr');
const settingsPath = path.resolve(appDataPath, 'settings.json');

let mainWindow;
let settings;

// Ensure dirs/files
createDirs.forEach((dir) => {
    fs.ensureDirSync(path.join(appDataPath, dir));
});

fs.ensureFileSync(path.join(appDataPath, 'models', 'Make sure all models are .onnx files'));

// Read settings
try {
    settings = fs.readJSONSync(settingsPath);
} catch (e) {
    console.error('Failed to read settings.json, using default settings.', e);

    settings = require('./main/default-settings');
    fs.writeJSONSync(settingsPath, settings);
}

// eslint-disable-next-line prefer-destructuring
settings = settings.settings[0];

// Initialize
const discordPresence = require('./main/discord-presence')(settings.rpc);
const BrowserWindow = require('./main/browser-window')(!settings.disableBlur);

remoteMain.initialize();

if (process.platform === 'win32') {
    app.setAppUserModelId('enhancr');
    app.setPath('userData', path.resolve(appDataPath, 'chromedata'));
}

app.once('ready', () => {
    let vibrancyOptions;
    let windowOptions = {
        width: 1024,
        height: 576,
        frame: false,
        show: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
        resizable: false,
        icon: path.join(__dirname, '/assets/enhancr-icon.png'),
        allowRunningInsecureContent: false,
        enableWebSQL: false,
    };

    // Platform-specific options
    // eslint-disable-next-line default-case
    switch (process.platform) {
        case 'win32': {
            windowOptions.transparent = true;

            if (settings.disableBlur) break;
            const isWin11 = os.release().split('.')[2] >= 22000;
            const effect = isWin11 ? 'acryllic' : 'blur';

            windowOptions = {
                ...windowOptions,
                vibrancy: {
                    theme: 'dark',
                    effect,
                    useCustomWindowRefreshMethod: false,
                    disableOnBlur: false,
                },
            };

            vibrancyOptions = {
                effect,
                disableOnBlur: false,
            };
            break;
        }
        case 'linux': {
            windowOptions.backgroundColor = '#333333';
            break;
        }
        case 'darwin': {
            windowOptions = {
                ...windowOptions,
                visualEffectState: 'active',
                backgroundColor: '#00000000',
                vibrancy: 'under-window',
                titleBarStyle: 'hiddenInset',
            };

            windowOptions.webPreferences.enableBlinkFeatures = 'AudioVideoTracks';
            break;
        }
    }

    mainWindow = new BrowserWindow(windowOptions);

    remoteMain.enable(mainWindow.webContents);
    setupIpc(mainWindow);

    if (vibrancyOptions) setVibrancy(mainWindow, vibrancyOptions);

    // Inject css for solid bg (for now)
    if (settings.disableBlur) {
        // Appear seamless when changing pages, but corners may be visible for a few frames.
        mainWindow.webContents.on('will-navigate', () => {
            mainWindow.setBackgroundColor('#242424');
        });

        // #1D1D1D = #242424 + rgb(28, 28, 28, .85)
        mainWindow.webContents.on('did-finish-load', () => {
            mainWindow.webContents.insertCSS('#light-mode { background: #1D1D1D !important }').then(() => {
                mainWindow.setBackgroundColor('#00000000');
            });
        });
    }

    mainWindow.once('show', () => {
        app.focus();
    });

    mainWindow.webContents.once('did-finish-load', () => {
        mainWindow.show();
        if (devMode) mainWindow.webContents.openDevTools({ mode: 'detach' });
    });

    // 'new-window' is deprecated; use setWindowOpenHandler
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.loadFile(path.join(__dirname, initialHtml));
});

app.once('window-all-closed', () => {
    const presenceStatus = discordPresence.client.status;
    const quit = () => app.quit();

    if (presenceStatus !== 'connected') {
        quit();
    } else {
        discordPresence.client.once('disconnect', quit);
        discordPresence.disconnect();

        setTimeout(() => {
            console.warn('Discord RPC still not disconnecting, forcing quit.');
            quit();
        }, 2000);
    }
});

app.on('second-instance', () => {
    app.focus();
});
