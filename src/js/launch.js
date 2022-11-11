const launchWrapper = document.querySelector('.launch-wrapper')
const progressBar = document.querySelector('.launch-progress')

var lightModeLayer = document.getElementById('light-mode');

if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches == false) {
    console.log('light mode detected')
    lightModeLayer.style.visibility = 'visible';
}

function launch() {
    launchWrapper.classList.add('animate__animated', 'animate__fadeOut');
    progressBar.classList.add('animate__animated', 'animate__fadeOutDown')
}

setTimeout(launch, 3450);


