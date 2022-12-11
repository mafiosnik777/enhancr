const { ipcRenderer } = require('electron');
const { app } = require('@electron/remote');

// Elements
// const lightModeLayer = document.getElementById('light-mode');
const createToggle = document.getElementById('hover-create-toggle');
const openToggle = document.getElementById('hover-open-toggle');

const versionText = document.getElementById('version');

const recentsContainer = document.getElementById('recents-container');
const recentItemTemplate = document.getElementById('recent-item-template');

/* window.addEventListener('error', (event) => {
    appLogo[0].style.animation = '';
}); */

/* if (!window.matchMedia('(prefers-color-scheme: dark)').matches) {
    console.log('light mode detected');
    lightModeLayer.style.visibility = 'visible';
} */

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

const recentProjects = parseJSON(localStorage.getItem('projects'), []);

if (navigator.userAgentData.platform === 'Linux') document.body.style.backgroundColor = '#333333';

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
    // path.basename alternative
    recentItem.querySelector('.project-title').textContent = project.split(/[\\/]/).pop();
    recentItem.querySelector('.project-path').textContent = project;

    recentItem.addEventListener('click', () => {
        loadProject(project);
    });

    recentsContainer.appendChild(clonedRecentItem);
});

// TODO: find a better way to get version without remote
versionText.textContent = versionText.textContent.replace('{{version}}', app.getVersion());
