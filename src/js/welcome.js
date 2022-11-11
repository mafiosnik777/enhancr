var path = require("path");
const { ipcRenderer } = require("electron");
const fs = require("fs");

const appDataPath = process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")

const appLogo = document.getElementsByClassName('app-icon');

var lightModeLayer = document.getElementById('light-mode');

window.addEventListener("error", (event) => {
  appLogo[0].style.animation = "";
});

if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches == false) {
  console.log('light mode detected')
  lightModeLayer.style.visibility = 'visible';
}

function getProject(index) {
  const json = fs.readFileSync(path.join(appDataPath, '/.enhancr/projects.json'), "utf8");
  const jsonData = JSON.parse(json);
  return jsonData[index];
}

function setProject(index, project) {
  const json = fs.readFileSync(path.join(appDataPath, '/.enhancr/projects.json'), "utf8");
  const jsonData = JSON.parse(json);
  jsonData[index] = project;
  fs.writeFileSync(path.join(appDataPath, '/.enhancr/projects.json'), JSON.stringify(jsonData));
}

var createToggle = document.getElementById("hover-create-toggle");
var openToggle = document.getElementById("hover-open-toggle");

function createToggleHover() {
  createToggle.style.background = "rgba(60, 60, 60, 0.3)";
  openToggle.style.background = "none";
}

createToggle.addEventListener("mouseover", createToggleHover);

function openToggleHover() {
  openToggle.style.background = "rgba(60, 60, 60, 0.3)";
  createToggle.style.background = "none";
}

openToggle.addEventListener("mouseover", openToggleHover);

var body = document.querySelector("body");

if (process.platform == "linux") {
  body.style.backgroundColor = "#333333";
}

// send ipc request from renderer to main process (create project)
function createProject() {
  ipcRenderer.send("create-project");
}
createToggle.addEventListener("click", createProject);

// Handle event reply from main process
ipcRenderer.on("project", (event, project) => {
  sessionStorage.setItem("currentProject", project);
  if (getProject("recent0") == "") {
    setProject("recent0", project);
  } else if (getProject("recent1") == "" && getProject("recent1") !== project) {
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent2") == "" && getProject("recent2") !== project) {
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent3") == "" && getProject("recent3") !== project) {
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent4") == "" && getProject("recent4") !== project) {
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent5") == "" && getProject("recent5") !== project) {
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent6") == "" && getProject("recent6") !== project) {
    setProject("recent6", getProject("recent5"));
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else if (getProject("recent7") == "" && getProject("recent7") !== project) {
    setProject("recent7", getProject("recent6"));
    setProject("recent6", getProject("recent5"));
    setProject("recent6", getProject("recent5"));
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", project);
  } else {
    console.log("Project already in recents.")
  }
});

// send ipc request from renderer to main process (open project)
function openProject() {
  ipcRenderer.send("open-project");
}
openToggle.addEventListener("click", openProject);

// Handle event reply from main process
ipcRenderer.on("openproject", (event, openproject) => {
  sessionStorage.setItem("currentProject", openproject);
  if (getProject("recent0") == "") {
    setProject("recent0", openproject);
  } else if (getProject("recent1") == "" && getProject("recent1") !== openproject) {
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent2") == "" && getProject("recent2") !== openproject) {
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent3") == "" && getProject("recent3") !== openproject) {
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent4") == "" && getProject("recent4") !== openproject) {
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent5") == "" && getProject("recent5") !== openproject) {
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent6") == "" && getProject("recent6") !== openproject) {
    setProject("recent6", getProject("recent5"));
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else if (getProject("recent7") == "" && getProject("recent7") !== openproject) {
    setProject("recent7", getProject("recent6"));
    setProject("recent6", getProject("recent5"));
    setProject("recent6", getProject("recent5"));
    setProject("recent5", getProject("recent4"));
    setProject("recent4", getProject("recent3"));
    setProject("recent3", getProject("recent2"));
    setProject("recent2", getProject("recent1"));
    setProject("recent1", getProject("recent0"));
    setProject("recent0", openproject);
  } else {
    console.log("Project already in recents.")
  }
  window.location.replace("../app.html");
});

// recents
var recent0 = document.getElementById("recent-0"),
  recent1 = document.getElementById("recent-1"),
  recent2 = document.getElementById("recent-2"),
  recent3 = document.getElementById("recent-3"),
  recent4 = document.getElementById("recent-4"),
  recent5 = document.getElementById("recent-5"),
  recent6 = document.getElementById("recent-6"),
  recent7 = document.getElementById("recent-7");

var title0 = document.getElementById("title0"),
  title1 = document.getElementById("title1"),
  title2 = document.getElementById("title2"),
  title3 = document.getElementById("title3"),
  title4 = document.getElementById("title4"),
  title5 = document.getElementById("title5"),
  title6 = document.getElementById("title6"),
  title7 = document.getElementById("title7");

var path0 = document.getElementById("path0"),
  path1 = document.getElementById("path1"),
  path2 = document.getElementById("path2"),
  path3 = document.getElementById("path3"),
  path4 = document.getElementById("path4"),
  path5 = document.getElementById("path5"),
  path6 = document.getElementById("path6"),
  path7 = document.getElementById("path7");

var noRecents = document.getElementById("no-recents");

if (getProject("recent0") == "") {
  recent0.style.visibility = "hidden";
} else {
  recent0.style.visibility = "visible";
  noRecents.style.visibility = "hidden";
  title0.textContent = path.basename(getProject("recent0"), ".enhncr");
  if (getProject("recent0").length >= 30) {
    path0.textContent = "../" + path.basename(path.dirname(getProject("recent0"))) + "/" + path.basename(getProject("recent0"));
  } else {
    path0.textContent = getProject("recent0");
  }
}

if (getProject("recent1") == "") {
  recent1.style.visibility = "hidden";
} else {
  recent1.style.visibility = "visible";
  title1.textContent = path.basename(getProject("recent1"), ".enhncr");
  if (getProject("recent1").length >= 30) {
    path1.textContent = "../" + path.basename(path.dirname(getProject("recent1"))) + "/" + path.basename(getProject("recent1"));
  } else {
    path1.textContent = getProject("recent1");
  }
};

if (getProject("recent2") == "") {
  recent2.style.visibility = "hidden";
} else {
  recent2.style.visibility = "visible";
  title2.textContent = path.basename(getProject("recent2"), ".enhncr");
  if (getProject("recent2").length >= 30) {
    path2.textContent = "../" + path.basename(path.dirname(getProject("recent2"))) + "/" + path.basename(getProject("recent2"));
  } else {
    path2.textContent = getProject("recent2");
  }
}

if (getProject("recent3") == "") {
  recent3.style.visibility = "hidden";
} else {
  recent3.style.visibility = "visible";
  title3.textContent = path.basename(getProject("recent3"), ".enhncr");
  if (getProject("recent3").length >= 30) {
    path3.textContent = "../" + path.basename(path.dirname(getProject("recent3"))) + + "/" + path.basename(getProject("recent3"));
  } else {
    path3.textContent = getProject("recent3");
  }
}

if (getProject("recent4") == "") {
  recent4.style.visibility = "hidden";
} else {
  recent4.style.visibility = "visible";
  title4.textContent = path.basename(getProject("recent4"), ".enhncr");
  if (getProject("recent4").length >= 30) {
    path4.textContent = "../" + path.basename(path.dirname(getProject("recent4"))) + "/" + path.basename(getProject("recent4"));
  } else {
    path4.textContent = getProject("recent4");
  }
}

if (getProject("recent5") == "") {
  recent5.style.visibility = "hidden";
} else {
  recent5.style.visibility = "visible";
  title5.textContent = path.basename(getProject("recent5"), ".enhncr");
  if (getProject("recent0").length >= 30) {
    path5.textContent = "../" + path.basename(path.dirname(getProject("recent5"))) + "/" + path.basename(getProject("recent5"));
  } else {
    path5.textContent = getProject("recent5");
  }
}

if (getProject("recent6") == "") {
  recent6.style.visibility = "hidden";
} else {
  recent6.style.visibility = "visible";
  title6.textContent = path.basename(getProject("recent6"), ".enhncr");
  if (getProject("recent0").length >= 30) {
    path6.textContent = "../" + path.basename(path.dirname(getProject("recent6"))) + "/" + path.basename(getProject("recent6"));
  } else {
    path6.textContent = getProject("recent6");
  }
}

if (getProject("recent7") == "") {
  recent7.style.visibility = "hidden";
} else {
  recent7.style.visibility = "visible";
  title7.textContent = path.basename(getProject("recent7"), ".enhncr");
  if (getProject("recent7").length >= 30) {
    path7.textContent = "../" + path.basename(path.dirname(getProject("recent0"))) + "/" + path.basename(getProject("recent7"));
  } else {
    path7.textContent = getProject("recent7");
  }
}

setTimeout(function () {
  appLogo[0].classList.remove('animate__fadeInDown');
}, 1000);

function load0() {
  sessionStorage.setItem("currentProject", getProject("recent0"));
  window.location.replace("../app.html");
}
function load1() {
  sessionStorage.setItem("currentProject", getProject("recent1"));
  window.location.replace("../app.html");
}
function load2() {
  sessionStorage.setItem("currentProject", getProject("recent2"));
  window.location.replace("../app.html");
}
function load3() {
  sessionStorage.setItem("currentProject", getProject("recent3"));
  window.location.replace("../app.html");
}
function load4() {
  sessionStorage.setItem("currentProject", getProject("recent4"));
  window.location.replace("../app.html");
}
function load5() {
  sessionStorage.setItem("currentProject", getProject("recent5"));
  window.location.replace("../app.html");
}
function load6() {
  sessionStorage.setItem("currentProject", getProject("recent6"));
  window.location.replace("../app.html");
}
function load7() {
  sessionStorage.setItem("currentProject", getProject("recent7"));
  window.location.replace("../app.html");
}

recent0.addEventListener("click", load0);
recent1.addEventListener("click", load1);
recent2.addEventListener("click", load2);
recent3.addEventListener("click", load3);
recent4.addEventListener("click", load4);
recent5.addEventListener("click", load5);
recent6.addEventListener("click", load6);
recent7.addEventListener("click", load7);

