const sysinfo = require('systeminformation');

const bytesToMegaBytes = bytes => bytes / (1024 ** 2);

async function setInfo() {
    var osInfo = sysinfo.osInfo().then(data => localStorage.setItem('os', `OS: ${data.distro} ${data.arch}`));
    var cpuInfo = sysinfo.cpu().then(data => localStorage.setItem('cpu', `CPU: ${data.manufacturer} ${data.brand} | [${data.physicalCores}C/${data.cores}T]`));
    var memInfo = sysinfo.mem().then(data => localStorage.setItem('ram', `RAM: ${Math.round(bytesToMegaBytes(data.total))} MB`));
    var gpuInfo = sysinfo.graphics().then(data => {
        // check for gpus and set final gpu based on hierarchy
        for (let i = 0; i < data.controllers.length; i++) {
          if (data.controllers[i].vendor.includes("Intel")) {
            localStorage.setItem('gpu', `GPU: ${data.controllers[i].model} | ${data.controllers[i].vram} MB`);
          }
          if (data.controllers[i].vendor.includes("AMD") || data.controllers[i].vendor.includes("Advanced Micro Devices")) {
            localStorage.setItem('gpu', `GPU: ${data.controllers[i].model} | ${data.controllers[i].vram} MB`);
          }
          if (data.controllers[i].vendor.includes("NVIDIA")) {
            localStorage.setItem('gpu', `GPU: ${data.controllers[i].model} | ${data.controllers[i].vram} MB`);
          }
        }
      });
}
setInfo();