const terminal = document.getElementById("terminal-text");

terminal.textContent += "•─────────•°• enhancr •°•─────────•"
terminal.textContent += "\r\n" + localStorage.getItem('os');
terminal.textContent += "\r\n" + localStorage.getItem('cpu');
terminal.textContent += "\r\n" + localStorage.getItem('ram');
terminal.textContent += "\r\n" + localStorage.getItem('gpu');
terminal.textContent += "\r\nStatus: Ready\r\n•°•────────────────────────────────•°•";

// prevent dragging of terminal content
terminal.addEventListener('dragstart', function(event) {
    event.preventDefault();
});