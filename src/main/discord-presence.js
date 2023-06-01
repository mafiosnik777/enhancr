const { EasyPresence } = require('easy-presence');
const { ipcMain, app } = require('electron');

const appVersion = app.getVersion();

const osNames = {
    linux: 'Linux',
    win32: 'Windows',
    darwin: 'macOS',
};

const statusTxt = {
    interpolate: 'Interpolating',
    upscale: 'Upscaling',
    restore: 'Restoring',
};

const appVersionText = `\u30FBenhancr - ${appVersion}`;
const basePresenceData = {
    details: `${osNames[process.platform]} ${process.arch}${appVersionText}`,
    assets: {
        large_image: 'enhancr',
        large_text: 'enhancr',
        small_image: 'enhancr-file',
    },
    buttons: [
        {
            label: 'Visit on GitHub',
            url: 'https://github.com/mafiosnik777/enhancr',
        },
    ],
    timestamps: { start: new Date() },
};

const rpcClient = new EasyPresence('1046415937886228558');
let presenceData;
let presenceEnabled;

function setPresence(status, { 
    fps, engine, percentage, startDate
} = {}) {
    let newPresenceData = basePresenceData;

    if (status) {
        newPresenceData = {
            ...basePresenceData,
            details: statusTxt[status] + appVersionText,
            state: `${engine} - ${fps} fps - ${percentage}%`,
            assets: {
                large_image: status,
                large_text: statusTxt[status],
                small_image: 'enhancr',
                small_text: `enhancr - ${appVersion}`,
            },
            timestamps: { start: startDate },
        };
    }

    presenceData = newPresenceData;
    if (presenceEnabled) rpcClient.setActivity(newPresenceData);
}

rpcClient.on('connected', () => {
    console.log('Discord RPC initialized - User:', rpcClient.environment.user.username);
});

module.exports = (startupEnabled) => {
    let inferenceRunning = false
    let startDate;

    const activePresence = (status, options) => {
        if (!inferenceRunning) {
            inferenceRunning = true
            startDate = new Date()
        }
        
        setPresence(status, { ...options, startDate });
    }

    ipcMain.on('rpc-done', () => {
        console.log('Process completed: Resetting Discord Presence')
        setTimeout(() => {
            setPresence(null)}, 5000);
    });
    ipcMain.on('rpc-interpolation', (event, fps, engine, percentage) => {
        activePresence('interpolate', { fps, engine, percentage });
    });
    ipcMain.on('rpc-upscaling', (event, fps, engine, percentage) => {
        activePresence('upscale', { fps, engine, percentage });
    });
    ipcMain.on('rpc-restoration', (event, fps, engine, percentage) => {
        activePresence('restore', { fps, engine, percentage });
    });

    if (startupEnabled) {
        presenceEnabled = true;
        setPresence(null);
    }

    return {
        client: rpcClient,
        connect: () => {
            presenceEnabled = true;
            rpcClient.setActivity(presenceData);
        },
        disconnect: () => {
            rpcClient.disconnect();
            presenceEnabled = false;
        },
    };
};
