const { ipcMain, dialog, app } = require('electron');
const fs = require('fs-extra');

const projectTemplate = require('./project-template.json');

module.exports = (mainWindow) => {
    function showDirDialog(event, {
        title = 'Choose output path', buttonLabel = 'Select Directory', replyWith,
    }) {
        dialog.showOpenDialog(mainWindow, {
            title,
            buttonLabel,
            properties: ['openDirectory'],
        }).then((file) => {
            if (!file.canceled) event.reply(replyWith, file.filePaths[0]);
        }).catch(console.error);
    }

    // Project create/open
    ipcMain.on('create-project', (event) => {
        const saveDialog = dialog.showSaveDialog(mainWindow, {
            title: 'Create new project',
            defaultPath: 'Untitled',
            filters: [{ name: 'enhancr Project File', extensions: ['enhncr'] }],
        });

        saveDialog.then((saveTo) => {
            const newProjectPath = saveTo.filePath;

            if (newProjectPath === '') {
                console.log('Project creation aborted');
                return;
            }

            fs.writeJSON(newProjectPath, projectTemplate)
                .then(() => {
                    console.log('Project data is written to the file successfully');
                    event.reply('openproject', newProjectPath);
                }).catch((err) => {
                    console.error('Error writing project file', err);
                });
        }).catch(console.error);
    });

    ipcMain.on('open-project', (event) => {
        const openDialog = dialog.showOpenDialog(mainWindow, {
            title: 'Open enhancr project',
            filters: [{ name: 'enhancr Project File', extensions: ['enhncr'] }],
            properties: ['openFile'],
        });

        openDialog.then((result) => {
            const openProjectPath = result.filePaths[0];

            if (openProjectPath) event.reply('openproject', openProjectPath);
            else console.log('Project loading aborted');
        }).catch(console.error);
    });

    // Window state
    ipcMain.on('minimize-window', () => {
        mainWindow.minimize();
    });

    ipcMain.on('isAppPackaged', (event) => {
        event.reply('packaged', app.isPackaged);
    });

    ipcMain.on('close-window', () => {
        mainWindow.close();
    });

    ipcMain.on('file-request', (event) => {
        showDirDialog(event, { replyWith: 'file' });
    });

    ipcMain.on('file-request-up', (event) => {
        showDirDialog(event, { replyWith: 'file-up' });
    });

    ipcMain.on('file-request-res', (event) => {
        showDirDialog(event, { replyWith: 'file-res' });
    });

    ipcMain.on('temp-dialog', (event) => {
        showDirDialog(event, { title: 'Choose cache folder', replyWith: 'temp-dir' });
    });
};
