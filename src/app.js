const { app, shell, BrowserWindow, nativeTheme } = require('electron');
const vibe = require('@pyke/vibe');

const fs = require('fs-extra');
const path = require('path');
const os = require('os');

const { localStorage, sessionStorage } = require('electron-browser-storage');

const remoteMain = require('@electron/remote/main');
const setupIpc = require('./main/setup-ipc');

if (!app.requestSingleInstanceLock()) app.quit();

nativeTheme.themeSource = 'dark';

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

const appDataPath = path.resolve(appDataPaths[process.platform], '.enhancr');
const settingsPath = path.resolve(appDataPath, 'settings.json');
let settings;

// this only seems to work in packaged state
if (!devMode) app.setAsDefaultProtocolClient('enhancr')

const singleInstanceLock = app.requestSingleInstanceLock();
if (!singleInstanceLock) {
    app.quit();
} else {
    app.on("second-instance", (_, argv) => {
        console.log(`Instance lock triggered`);
        if (argv.length > 1) {
            // Only try this if there is an argv (might be redundant)
            if (process.platform == "win32" || process.platform === "linux") {
                try {
                    console.log(`${argv[argv.length - 1]}`);
                    BrowserWindow.getFocusedWindow().webContents.send('OAuth', argv[argv.length - 1])
                } catch {
                    console.log(
                        `Direct link to file - FAILED: ${argv}`,
                    );
                }
            }
        }
    });
}

// Ensure dirs/files
createDirs.forEach((dir) => {
    fs.ensureDirSync(path.join(appDataPath, dir));
});

fs.unlink(path.join(appDataPath, 'models', 'Make sure all models are .onnx files'));

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

let scale = () => {
    if (settings.uiScale == 0) return 1.00;
    else if (settings.uiScale == 25) return 1.15;
    else if (settings.uiScale == 50) return 1.25;
    else if (settings.uiScale == 75) return 1.35;
    else if (settings.uiScale == 100) return 1.45;
    else return 1.25;
}

app.commandLine.appendSwitch('high-dpi-support', scale())
app.commandLine.appendSwitch('force-device-scale-factor', scale())

// Initialize
vibe.setup(app);
const discordPresence = require('./main/discord-presence')(settings.rpc);

remoteMain.initialize();

if (process.platform === 'win32') {
    app.setAppUserModelId('enhancr');
    app.setPath('userData', path.resolve(appDataPath, 'chromedata'));
}

let frame = () => {
    if (vibe.platform.isWin11()) return true; else return false;
}

app.whenReady().then(async () => {

    let windowOptions = {
        width: 1024,
        height: 576,
        frame: frame(),
        show: false,
        autoHideMenuBar: true,
        transparent: false,
        backgroundColor: '#00000000',
        show: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            backgroundThrottling: false
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
            windowOptions = {
                ...windowOptions,
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
                transparency: true
            };

            windowOptions.webPreferences.enableBlinkFeatures = 'AudioVideoTracks';
            break;
        }
    }

    const mainWindow = new BrowserWindow(windowOptions);

    if (vibe.platform.isWin11()) vibe.applyEffect(mainWindow, 'mica'); else vibe.applyEffect(mainWindow, 'blurbehind');

    remoteMain.enable(mainWindow.webContents);
    setupIpc(mainWindow);

    // Inject css for solid bg (for now)
    if (settings.disableBlur) {
        // Appear seamless when changing pages, but corners may be visible for a few frames.
        mainWindow.webContents.on('will-navigate', () => {
            mainWindow.setBackgroundColor('#222222');
        });

        // #1D1D1D = #242424 + rgb(28, 28, 28, .85)
        mainWindow.webContents.on('did-finish-load', () => {
            mainWindow.webContents.insertCSS('#light-mode { background: #1D1D1D !important }').then(() => {
                mainWindow.setBackgroundColor('#222222');
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

    const patreonUntil = await localStorage.getItem('patreonUntil');
    const validUntil = new Date(patreonUntil || 0);
    const now = new Date();

    // check if app is packaged -> if not skip auth / if yes check if subscription is still valid
    if (!devMode && (patreonUntil == null || now > validUntil)) mainWindow.loadFile(path.join(__dirname, './pages/auth.html'))
    else mainWindow.loadFile(path.join(__dirname, './pages/welcome.html'))

    app.on('open-url', (event, url) => {
        console.log(`${url}`);
    })
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
