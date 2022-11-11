const sysinfo = require('systeminformation');

const bytesToMegaBytes = bytes => bytes / (1024 ** 2);

async function setInfo() {
    var osInfo = sysinfo.osInfo().then(data => localStorage.setItem('os', `OS: ${data.distro} ${data.arch}`));
    var cpuInfo = sysinfo.cpu().then(data => localStorage.setItem('cpu', `CPU: ${data.manufacturer} ${data.brand} | [${data.physicalCores}C/${data.cores}T]`));
    var memInfo = sysinfo.mem().then(data => localStorage.setItem('ram', `RAM: ${Math.round(bytesToMegaBytes(data.total))} MB`));
    var gpuInfo = sysinfo.graphics().then(data => localStorage.setItem('gpu', `GPU: ${data.controllers[0].model} | ${data.controllers[0].vram} MB`));
}
setInfo();