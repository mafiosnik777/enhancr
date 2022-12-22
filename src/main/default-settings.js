const path = require('path');
const os = require('os');

module.exports = {
    settings: [
        {
            preview: true,
            disableBlur: false,
            rpc: true,
            theme: 'blue',
            rifeTta: false,
            rifeUhd: false,
            sc: true,
            skip: false,
            cainSc: true,
            fp16: true,
            num_streams: 2,
            denoiseStrength: 20,
            deblockStrength: 15,
            tileRes: '512x512',
            tiling: false,
            temp: path.resolve(os.tmpdir(), 'enhancr'),
            shapeRes: '1080x1920',
            shapes: false,
            trimAccurate: false,
            hwEncode: false,
            sensitivityValue: 0.180,
            sensitivity: false,
            customModel: false,
            unsupportedEngines: false,
            systemPython: false,
            language: 'english',
        },
    ],
};
