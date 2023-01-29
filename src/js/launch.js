const launchWrapper = document.querySelector('.launch-wrapper');
const progressBar = document.querySelector('.launch-progress');

const os = require('os');

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

/* const lightModeLayer = document.getElementById('light-mode');

if (!window.matchMedia('(prefers-color-scheme: dark)').matches) {
    console.log('light mode detected')
    lightModeLayer.style.visibility = 'visible';
} */

setTimeout(() => {
    launchWrapper.classList.add('animate__animated', 'animate__fadeOut');
    progressBar.classList.add('animate__animated', 'animate__fadeOutDown');
}, 3450);

setTimeout(() => {
    window.location.replace('./auth.html');
}, 4500);
