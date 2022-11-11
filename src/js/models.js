const fse = require('fs-extra');
const path = require("path");
const { shell } = require('electron')

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

const customModelSpan = document.getElementById('custom-model');
const modelsHider = document.getElementById('models-hider'); 
const modelDropdown = document.getElementById('custom-model-dropdown');

function toggleCustomModelDropdown() {
    modelsHider.style.visibility = 'visible';
    modelDropdown.style.visibility = 'visible';

    // load custom models
    fse.readdir(path.join(appDataPath, '/.enhancr/models/RealESRGAN'), function (err, files) {
        //handling error
        if (err) {
            return console.log('Unable to scan directory: ' + err);
        }
        console.log('[enhancr] Custom Models found: ' + files.length);
        modelDropdown.innerHTML = '';
        files.forEach(function (file, i = 0) {
            let model = document.createElement('div');
            model.classList.add('optionModel');
            model.setAttribute('id', `model${i}`);
            modelDropdown.append(model);
            model.innerHTML = file;
            i++;
        });

        let modelsOptionItems = [].slice.call(document.getElementsByClassName('optionModel'));

        modelsOptionItems.forEach(function (option) {
            option.addEventListener('click', function () {
                const customModelText = document.getElementById('custom-model-text');
                customModelText.innerHTML = option.innerHTML;
                modelsHider.style.visibility = 'hidden';
                modelDropdown.style.visibility = 'hidden';
                sessionStorage.setItem('customModel', 'true');
                sessionStorage.setItem('customModelName', option.innerHTML);
            });
        });
    });
}
customModelSpan.addEventListener('click', toggleCustomModelDropdown);

function hideHider() {
    modelsHider.style.visibility = 'hidden';
    modelDropdown.style.visibility = 'hidden';
}
modelsHider.addEventListener('click', hideHider);