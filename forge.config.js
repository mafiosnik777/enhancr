const fs = require('fs-extra');
const path = require('path');
const sass = require('sass');

const assetBuildDir = path.resolve(__dirname, './build');

// const includePaths = [
//     'src',
//     'build',
//     'LICENSE',
//     'package.json',
//     'node_modules',
// ];

// TODO: resolve issues with packaging when ignoring paths 
// const ignoredPaths = fs.readdirSync(__dirname).filter((pathname) => (
//     !includePaths.includes(pathname)
// ));

// if (process.platform === 'win32') process.env.GYP_MSVS_VERSION = '2019';

module.exports = {
    packagerConfig: {
        icon: './src/assets/enhancr-icon',
        asar: {
            unpack: "**/node_modules/{@mafiosnik,@ffprobe-installer,@ffmpeg-installer,axios}/**/*"
          },
        ignore: [
            'src/scss',
            'external',
            'inference',
            'utils',
            /\.map$/i,
            // ...ignoredPaths,
        ],
    },
    hooks: {
        generateAssets: async () => {
            const compiledScss = sass.compile('./src/scss/app.scss', {
                style: 'compressed',
                sourceMap: true,
            });

            compiledScss.css += '\n/*# sourceMappingURL=app.min.css.map */';

            await fs.ensureDir(assetBuildDir);
            await fs.emptyDir(assetBuildDir);

            await fs.writeFile(
                path.resolve(assetBuildDir, 'app.min.css'),
                compiledScss.css,
            );
            await fs.writeFile(
                path.resolve(assetBuildDir, 'app.min.css.map'),
                JSON.stringify(compiledScss.sourceMap),
            );
        },
        postMake:  () => {
            const pythonDir = './src/external/'
            const inferenceDir = './src/inference/'
            const utilsDir = './src/utils/'
            const targetDirPython = './out/enhancr-win32-x64/resources/external'
            const targetDirInference = './out/enhancr-win32-x64/resources/inference'
            const targetDirUtils = './out/enhancr-win32-x64/resources/utils'
            fs.copySync(pythonDir, targetDirPython);
            fs.copySync(inferenceDir, targetDirInference);
            fs.copySync(utilsDir, targetDirUtils);
        }
    },
    makers: [
        {
            name: '@electron-forge/maker-squirrel',
            config: {
                name: 'enhancr',
            },
        },
        {
            name: '@electron-forge/maker-zip',
            platforms: ['darwin'],
        },
        {
            name: '@electron-forge/maker-deb',
            config: {},
        },
        {
            name: '@electron-forge/maker-rpm',
            config: {},
        },
    ],
};
