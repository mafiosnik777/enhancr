const launchWrapper = document.querySelector('.launch-wrapper');
const progressBar = document.querySelector('.launch-progress');

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
    window.location.replace('./welcome.html');
}, 4500);
