const fs = require('fs-extra');
const path = require('path');
const sass = require('sass');

const assetBuildDir = path.resolve(__dirname, './build');
const includePaths = [
    'src',
    'build',
    'LICENSE',
    'package.json',
    'node_modules',
];

const ignoredPaths = fs.readdirSync(__dirname).filter((pathname) => (
    !includePaths.includes(pathname)
));

if (process.platform === 'win32') process.env.GYP_MSVS_VERSION = '2017';

module.exports = {
    packagerConfig: {
        icon: './src/assets/enhancr-icon',
        asar: false,
        ignore: [
            'src/scss',
            ...ignoredPaths,
        ],
    },
    hooks: {
        generateAssets: async () => {
            await fs.ensureDir(assetBuildDir);
            await fs.emptyDir(assetBuildDir);

            await fs.writeFile(
                path.resolve(assetBuildDir, 'app.min.css'),
                sass.compile('./src/scss/app.scss', { style: 'compressed', sourceMap: true }).css,
            );
        },
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
