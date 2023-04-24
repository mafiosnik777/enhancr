const { ipcRenderer } = require('electron');
const { app } = require('@electron/remote');
const os = require('os');
const path = require('path');
const fs = require('fs-extra');

// Elements
const lightModeLayer = document.getElementById('light-mode');
const createToggle = document.getElementById('hover-create-toggle');
const openToggle = document.getElementById('hover-open-toggle');

const versionText = document.getElementById('version');

const recentsContainer = document.getElementById('recents-container');
const recentItemTemplate = document.getElementById('recent-item-template');

/* window.addEventListener('error', (event) => {
    appLogo[0].style.animation = '';
}); */

var winControls = document.getElementById("win-controls");

//Window controls
const isWin11 = os.release().split('.')[2] >= 22000;
if (process.platform == "win32" && !isWin11 || process.platform == "linux") {
    winControls.style.visibility = "visible";
    document.getElementById('patreon-profile').style.top = '8%';
    document.getElementById('profile-container').style.top = '9.3%';
    document.getElementById('close').style.top = '1%';
    document.getElementById('minimize').style.top = '2%';

    // Window controls
    const minimize = document.getElementById("minimize");
    minimize.addEventListener("click", function () {
        ipcRenderer.send("minimize-window");
    });
    const close = document.getElementById("close");
    close.addEventListener("click", function () {
        ipcRenderer.send("close-window");
    });
} else {
    winControls.style.visibility = "hidden";
}

if (!window.matchMedia('(prefers-color-scheme: dark)').matches) {
    console.log('light mode detected');
    lightModeLayer.style.visibility = 'visible';
}

function parseJSON(str, defaults) {
    if (!str) return defaults;

    try {
        return JSON.parse(str);
    } catch {
        return defaults;
    }
}

function loadProject(projectPath) {
    sessionStorage.setItem('currentProject', projectPath);
    window.location.replace('../app.html');
}

const appDataPaths = {
    win32: process.env.APPDATA,
    darwin: `${process.env.HOME}/Library/Preferences`,
    linux: `${process.env.HOME}/.local/share`,
};
const appDataPath = path.resolve(appDataPaths[process.platform], '.enhancr');

if (!(localStorage.getItem('patreonUser') == null)) {
// set users profile pic
let profilePic = path.join(appDataPath, 'profile.jpeg');
document.getElementById('patreon-profile').src = profilePic;

// change profile name
document.getElementById('profile-span').textContent = localStorage.getItem('patreonUser')

//change tier color
if (localStorage.getItem('Tier') == '1000') {
    document.getElementById('tier-icon').style.color = '#c69f31'
} else {
    document.getElementById('tier-icon').style.color = 'silver'
}
} else {
    document.getElementById('patreon-profile').style.visibility = 'hidden';
    document.getElementById('profile-container').style.visibility = 'hidden';
}

const recentProjects = parseJSON(localStorage.getItem('projects'), []);

const blurLayer = document.getElementById('light-mode')
const border = document.getElementById('win10-border');

// change blur on win 11/10
let winOsBuild = parseInt(os.release().split(".")[2]);
if (winOsBuild >= 22000 && process.platform == 'win32') {
    blurLayer.style.visibility = 'hidden';
    border.style.visibility = 'hidden';
} else {
    blurLayer.style.visibility = 'visible';
    border.style.visibility = 'visible';
}

// Button functionality
createToggle.addEventListener('click', () => {
    ipcRenderer.send('create-project');
});

openToggle.addEventListener('click', () => {
    ipcRenderer.send('open-project');
});

function createToggleHover() {
    createToggle.style.background = "rgba(60, 60, 60, 0.3)";
    openToggle.style.background = "none";
}

createToggle.addEventListener("mouseover", createToggleHover);

function openToggleHover() {
    openToggle.style.background = "rgba(60, 60, 60, 0.3)";
    createToggle.style.background = "none";
}

openToggle.addEventListener("mouseover", openToggleHover);

// Received from ipcMain dialog
ipcRenderer.on('openproject', (event, project) => {
    // Push to top if opened recently
    if (recentProjects.includes(project)) {
        recentProjects.splice(recentProjects.indexOf(project), 1);
    }

    recentProjects.unshift(project);

    // Remove projects from history past 7 files (for now)
    if (recentProjects.length > 7) recentProjects.splice(8);

    localStorage.setItem('projects', JSON.stringify(recentProjects));
    loadProject(project);
});

recentProjects.forEach((project, i) => {
    const clonedRecentItem = recentItemTemplate.content.cloneNode(true);
    const recentItem = clonedRecentItem.querySelector('.recent');
    const delay = 1 + (i * 0.3);

    recentItem.style.animationDelay = `${delay}s`;

    recentItem.querySelector('.project-title').textContent = path.parse(project).name;
    if(project.length >= 30) {
        recentItem.querySelector('.project-path').textContent = "../" + path.basename(path.dirname(project)) + "/" + path.basename(project);
    } else {
        recentItem.querySelector('.project-path').textContent = project;
    }
    

    recentItem.addEventListener('click', () => {
        loadProject(project);
    });
    // only render recentItem if project still exists
    if (fs.existsSync(project)) recentsContainer.appendChild(clonedRecentItem);
});

// TODO: find a better way to get version without remote
versionText.textContent = versionText.textContent.replace('{{version}}', app.getVersion());
