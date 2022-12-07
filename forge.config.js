if (process.platform === 'win32') process.env.GYP_MSVS_VERSION = '2017';

module.exports = {
    packagerConfig: {
        icon: './src/assets/enhancr-icon',
        asar: false,
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
