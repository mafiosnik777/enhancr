const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const fse = require("fs-extra");
const find = require('find-process');
const os = require('os');

const electron = require('electron');

var colors = require('colors');

electron.app.commandLine.appendSwitch("enable-transparent-visuals");

const remoteMain = require('@electron/remote/main');
remoteMain.initialize();

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share");

// Skip loading screen (dev mode)
var devMode = true;

if (process.platform == "win32") {
  app.setAppUserModelId("enhancr");
}

function getOSInfo() {
  if (process.platform == "linux") {
    return "Linux";
  }
  if (process.platform == "win32") {
    return "Windows";
  }
  if (process.platform == "darwin") {
    return "macOS";
  }
}

// create paths if not existing
if (!fse.existsSync(path.join(appDataPath, '/.enhancr'))) {
  fse.mkdirSync(path.join(appDataPath, '/.enhancr'));
};
if (!fse.existsSync(path.join(appDataPath, '/.enhancr/models'))) {
  fse.mkdirSync(path.join(appDataPath, '/.enhancr/models'));
};
if (!fse.existsSync(path.join(appDataPath, '/.enhancr/models/RealESRGAN'))) {
  fse.mkdirSync(path.join(appDataPath, '/.enhancr/models/RealESRGAN'));
};
if (!fse.existsSync(path.join(appDataPath, '/.enhancr/models/engine'))) {
  fse.mkdirSync(path.join(appDataPath, '/.enhancr/models/engine'));
};
if (!fse.existsSync(path.join(appDataPath, '/.enhancr/thumbs'))) {
  fse.mkdirSync(path.join(appDataPath, '/.enhancr/thumbs'));
};
if (!fse.existsSync(path.join(appDataPath, '/.enhancr/models/', 'Make sure all models are .onnx files'))) {
  fse.writeFile(path.join(appDataPath, '/.enhancr/models/', 'Make sure all models are .onnx files'), '');
};

// create projects storage
if (!fs.existsSync(path.join(appDataPath, '/.enhancr/projects.json'))) {
  var projects = {
    recent0: "",
    recent1: "",
    recent2: "",
    recent3: "",
    recent4: "",
    recent5: "",
    recent6: "",
    recent7: ""
  };
  var data = JSON.stringify(projects);
  fs.writeFile(path.join(appDataPath, '/.enhancr/projects.json'), data, (err) => {
    if (err) {
      console.log("Error writing file", err);
    } else {
      console.log("JSON data is written to the file successfully");
    }
  });
};

// create settings file
if (!fs.existsSync(path.join(appDataPath, '/.enhancr/settings.json'))) {
  var settings = {
    settings: [
      {
        preview: true,
        disableBlur: false,
        rpc: true,
        theme: 'blue',
        rifeTta: false,
        rifeUhd: false,
        rifeSc: true,
        cainSc: true,
        fp16: true,
        num_streams: 2,
        denoiseStrength: 20,
        deblockStrength: 15,
        tileRes: "512x512",
        tiling: false,
        shapeRes: "1080x1920",
        shapes: false,
        trimAccurate: false,
        customModel: false,
        systemPython: false,
        language: "english"
      },
    ],
  };
  var data = JSON.stringify(settings);
  fs.writeFile(path.join(appDataPath, '/.enhancr/settings.json'), data, (err) => {
    if (err) {
      console.log("Error writing file", err);
    } else {
      console.log("JSON data is written to the file successfully");
    }
  });
};

var osInfo = getOSInfo();

fs.readFile(path.join(appDataPath, '/.enhancr/settings.json'), (err, settings) => {
  if (err) throw err;
  let json = JSON.parse(settings);

  if (json.settings[0].disableBlur == false) {
    // discord rpc
    var client = new (require("easy-presence").EasyPresence)("1046415937886228558");
    client.on("connected", () => {
      console.log("discord rpc initialized - user: ", client.environment.user.username);
    });
    client.setActivity({
      details: osInfo + " " + process.arch + "・enhancr - 0.9.1",
      assets: {
        large_image: "enhancr",
        large_text: "enhancr",
        small_image: "enhancr-file"
      },
      buttons: [
        {
          label: "Visit on GitHub",
          url: "https://github.com/mafiosnik777/enhancr"
        }
      ],
      timestamps: { start: new Date() }
    });
  }})


// window creation

if (process.platform == "linux") {
  // Create the browser window on linux.
  createWindow = () => {
    const mainWindow = new BrowserWindow({
      width: 1024,
      height: 576,
      frame: false,
      titleBarStyle: "hiddenInset",
      webPreferences: {
        nodeIntegration: true,
        nodeIntegrationInWorker: true,
        nodeIntegrationInSubFrames: true,
        enableRemoteModule: true,
        contextIsolation: false,
        plugins: true,
        backgroundThrottling: false
      },
      resizable: false,
      backgroundColor: "#333333",
      visualEffectState: "active",
    });

    remoteMain.enable(mainWindow.webContents);
    mainWindow.setIcon(path.join(__dirname, '/assets/enhancr-icon.png'));

    // Prevent links from loading in app when clicked
    mainWindow.webContents.on("new-window", function (e, url) {
      e.preventDefault();
      require("electron").shell.openExternal(url);
    });

    // load initial html file
    let initialHtml = "./pages/launch.html";
    if (devMode == true) {
      initialHtml = "./pages/welcome.html";
      // Open Dev Tools at launch (detached)
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
    mainWindow.loadFile(path.join(__dirname, initialHtml));
    console.log("\x1b[32m✔\x1b[0m" + " Sucessfully loaded " + initialHtml);

    // Swap out windows (dev mode disabled)
    function swap() {
      mainWindow.loadFile(path.join(__dirname, "./pages/welcome.html"));
      console.log(
        "\x1b[32m✔\x1b[0m" + " Initialization done! Swapping out windows."
      );
    }
    if (devMode == false) {
      setTimeout(swap, 4500);
    }
    // fire event when minimize button is clicked on win32/linux
    ipcMain.on("minimize-window", function (event) {
      mainWindow.minimize();
    });
    // fire event when close button is clicked on win32/linux
    ipcMain.on("close-window", function (event) {
      mainWindow.close();
    });
  };
}

// Create the browser window on Windows.
if (process.platform == "win32") {
  createWindow = () => {
    // change blur on windows 10/11
    let winOsBuild = parseInt(os.release().split(".")[2]);
    function getEffect() {
      if (winOsBuild >= 22000) {
        return 'acrylic';
      } else {
        return 'blur';
      }
    }
    let effect = getEffect();

    function getTransparency() {
      if (winOsBuild >= 22000) {
        return false;
      } else {
        return true;
      }
    }
    let transparency = getTransparency()

    let vibrancy = {
      theme: 'dark',
      effect: effect,
      useCustomWindowRefreshMethod: false,
      disableOnBlur: false,
      debug: false,
    }

    fs.readFile(path.join(appDataPath, '/.enhancr/settings.json'), (err, settings) => {
      if (err) throw err;
      let json = JSON.parse(settings);

      if (json.settings[0].disableBlur == false) {
        const { BrowserWindow } = require("electron-acrylic-window");
        // Create the browser window.
        mainWindow = new BrowserWindow({
          width: 1024,
          height: 576,
          frame: false,
          titleBarStyle: "hiddenInset",
          webPreferences: {
            nodeIntegration: true,
            nodeIntegrationInWorker: true,
            nodeIntegrationInSubFrames: true,
            enableRemoteModule: true,
            contextIsolation: false,
            plugins: true,
            backgroundThrottling: false
          },
          resizable: false,
          transparent: transparency,
          vibrancy: vibrancy,
          visualEffectState: "active",
        });
        mainWindow.blurType = effect;
        mainWindow.setBlur = "false";
        const { setVibrancy } = require("electron-acrylic-window");
        setVibrancy(mainWindow, { effect: effect, disableOnBlur: false });

        remoteMain.enable(mainWindow.webContents);
        mainWindow.setIcon(path.join(__dirname, '/assets/enhancr-icon.png'));

        // Prevent links from loading in app when clicked
        mainWindow.webContents.on("new-window", function (e, url) {
          e.preventDefault();
          require("electron").shell.openExternal(url);
        });

        // load initial html file
        let initialHtml = "./pages/launch.html";
        if (devMode == true) {
          initialHtml = "./pages/welcome.html";
          // Open Dev Tools at launch (detached)
          mainWindow.webContents.openDevTools({ mode: "detach" });
        }
        mainWindow.loadFile(path.join(__dirname, initialHtml));
        console.log("✔".green + " Sucessfully loaded " + initialHtml);

        // Swap out windows (dev mode disabled)
        function swap() {
          mainWindow.loadFile(path.join(__dirname, "./pages/welcome.html"));
          console.log(
            "✔".green + " Initialization done! Swapping out windows."
          );
        }
        if (devMode == false) {
          setTimeout(swap, 4500);
        }
        // fire event when minimize button is clicked on win32/linux
        ipcMain.on("minimize-window", function (event) {
          mainWindow.minimize();
        });
        // fire event when close button is clicked on win32/linux
        ipcMain.on("close-window", function (event) {
          mainWindow.close();
        });
      } else {
        const { BrowserWindow } = require('electron');
        mainWindow = new BrowserWindow({
          width: 1024,
          height: 576,
          frame: false,
          titleBarStyle: "hiddenInset",
          webPreferences: {
            nodeIntegration: true,
            nodeIntegrationInWorker: true,
            nodeIntegrationInSubFrames: true,
            enableRemoteModule: true,
            contextIsolation: false,
            plugins: true,
            backgroundThrottling: false
          },
          resizable: false,
          transparent: transparency,
          backgroundColor: '#242424',
          visualEffectState: "active"
        });
        remoteMain.enable(mainWindow.webContents);
        mainWindow.setIcon(path.join(__dirname, '/assets/enhancr-icon.png'));

        // Prevent links from loading in app when clicked
        mainWindow.webContents.on("new-window", function (e, url) {
          e.preventDefault();
          require("electron").shell.openExternal(url);
        });

        // load initial html file
        let initialHtml = "./pages/launch.html";
        if (devMode == true) {
          initialHtml = "./pages/welcome.html";
          // Open Dev Tools at launch (detached)
          mainWindow.webContents.openDevTools({ mode: "detach" });
        }
        mainWindow.loadFile(path.join(__dirname, initialHtml));
        console.log("✔".green + " Sucessfully loaded " + initialHtml);

        // Swap out windows (dev mode disabled)
        function swap() {
          mainWindow.loadFile(path.join(__dirname, "./pages/welcome.html"));
          console.log(
            "✔".green + " Initialization done! Swapping out windows."
          );
        }
        if (devMode == false) {
          setTimeout(swap, 4500);
        }
        // fire event when minimize button is clicked on win32/linux
        ipcMain.on("minimize-window", function (event) {
          mainWindow.minimize();
        });
        // fire event when close button is clicked on win32/linux
        ipcMain.on("close-window", function (event) {
          mainWindow.close();
        });
      }
    });
  };
}

// Create the browser window on macOS.
if (process.platform == "darwin") {
  createWindow = () => {
    const mainWindow = new BrowserWindow({
      width: 1024,
      height: 576,
      frame: false,
      titleBarStyle: "hiddenInset",
      webPreferences: {
        nodeIntegration: true,
        nodeIntegrationInWorker: true,
        nodeIntegrationInSubFrames: true,
        enableRemoteModule: true,
        contextIsolation: false,
        enableBlinkFeatures: 'AudioVideoTracks',
        backgroundThrottling: false
      },
      resizable: false,
      transparent: true,
      transparency: true,
      backgroundColor: "#00000000",
      vibrancy: "under-window",
      visualEffectState: "active",
    });

    remoteMain.enable(mainWindow.webContents);
    mainWindow.setIcon(path.join(__dirname, '/assets/enhancr-icon.png'));

    // Prevent links from loading in app when clicked
    mainWindow.webContents.on("new-window", function (e, url) {
      e.preventDefault();
      require("electron").shell.openExternal(url);
    });

    // load initial html file
    let initialHtml = "./pages/launch.html";
    if (devMode == true) {
      initialHtml = "./pages/welcome.html";
      // Open Dev Tools at launch (detached)
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
    mainWindow.loadFile(path.join(__dirname, initialHtml));
    console.log("\x1b[32m✔\x1b[0m" + " Sucessfully loaded " + initialHtml);

    // Swap out windows (dev mode disabled)
    function swap() {
      mainWindow.loadFile(path.join(__dirname, "./pages/welcome.html"));
      console.log(
        "\x1b[32m✔\x1b[0m" + " Initialization done! Swapping out windows."
      );
    }
    if (devMode == false) {
      setTimeout(swap, 4500);
    }
  };
}

ipcMain.on("file-request", (event) => {
  // If the platform is 'win32' or 'Linux'
  if (process.platform !== "darwin") {
    // Resolves to a Promise<Object>
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Specifying the File Selector Property
        properties: ["openDirectory"],
      })
      .then((file) => {
        // Stating whether dialog operation was
        // cancelled or not.
        console.log(file.canceled);
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          console.log(filepath);
          event.reply("file", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    // If the platform is 'darwin' (macOS)
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Selector Property In macOS
        properties: ["openDirectory"],
      })
      .then((file) => {
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          event.reply("file", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }
});

ipcMain.on("file-request-up", (event) => {
  // If the platform is 'win32' or 'Linux'
  if (process.platform !== "darwin") {
    // Resolves to a Promise<Object>
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Specifying the File Selector Property
        properties: ["openDirectory"],
      })
      .then((file) => {
        // Stating whether dialog operation was
        // cancelled or not.
        console.log(file.canceled);
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          console.log(filepath);
          event.reply("file-up", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    // If the platform is 'darwin' (macOS)
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Selector Property In macOS
        properties: ["openDirectory"],
      })
      .then((file) => {
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          event.reply("file-up", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }
});

ipcMain.on("file-request-res", (event) => {
  // If the platform is 'win32' or 'Linux'
  if (process.platform !== "darwin") {
    // Resolves to a Promise<Object>
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Specifying the File Selector Property
        properties: ["openDirectory"],
      })
      .then((file) => {
        // Stating whether dialog operation was
        // cancelled or not.
        console.log(file.canceled);
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          console.log(filepath);
          event.reply("file-res", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    // If the platform is 'darwin' (macOS)
    dialog
      .showOpenDialog({
        title: "Choose output path",
        buttonLabel: "Select Directory",
        // Selector Property In macOS
        properties: ["openDirectory"],
      })
      .then((file) => {
        if (!file.canceled) {
          const filepath = file.filePaths[0].toString();
          event.reply("file-res", filepath);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }
});

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
if (process.platform == "darwin" || process.platform == "win32" || process.platform == "linux") {
  app.on("ready", createWindow);
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", async () => {
  if (process.platform !== "darwin") {
    find('name', 'VSPipe', false).then(function (list) {
      var i;
      for (i = 0; i < list.length; i++) {
        process.kill(list[i].pid);
      }
    });
    app.quit();
    return;
  }
});

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.

// Create new project
var currentWindow = BrowserWindow.getFocusedWindow();
ipcMain.on("create-project", function (event) {
  var options = {
    title: "Create new project",
    filters: [{ name: "Project", extensions: ["enhncr"] }],
  };
  let saveDialog = dialog.showSaveDialog(currentWindow, options);
  saveDialog
    .then(function (saveTo) {
      const newProjectPath = saveTo.filePath;
      // specify project save structure
      var project = {
        interpolation: [
          {
            inputFile: "",
            outputPath: "",
            codec: "H264",
            outputContainer: "",
            engine: "",
            model: "",
            params: ""
          },
        ],
        upscaling: [
          {
            inputFile: "",
            outputPath: "",
            codec: "H264",
            outputContainer: "",
            engine: "",
            scale: "",
            params: ""
          },
        ],
        restoration: [
          {
            inputFile: "",
            outputPath: "",
            codec: "H264",
            outputContainer: "",
            engine: "",
            model: "",
            params: ""
          },
        ]
      };
      var data = JSON.stringify(project);
      fs.writeFile(newProjectPath, data, (err) => {
        if (err) {
          console.log("Error writing project file", err);
        } else {
          console.log("Project data is written to the file successfully");
        }
      });
      if (newProjectPath === "") {
        console.log("Project creation aborted");
      } else {
        event.reply("project", newProjectPath);
        BrowserWindow.getFocusedWindow().loadFile(
          path.join(__dirname, "./app.html")
        );
      }
    })
    .catch((err) => {
      console.log(err);
    });
});

ipcMain.on("open-project", function (event) {
  dialog
    .showOpenDialog({
      title: "Open enhancr project",
      filters: [{ name: "Enhancr Project", extensions: ["enhncr"] }],
      properties: ["openFile"],
    })
    .then((result) => {
      const openProjectPath = result.filePaths[0];
      if (openProjectPath === undefined) {
        console.log("Project loading aborted");
      } else {
        event.reply("openproject", openProjectPath);
      }
    })
    .catch((err) => {
      console.log(err);
    });
});
