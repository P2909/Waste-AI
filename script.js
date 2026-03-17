const button = document.getElementById("helloButton");
const message = document.getElementById("message");

if (button && message) {
  button.addEventListener("click", () => {
    const now = new Date().toLocaleTimeString();
    message.textContent = `Hello from Waste AI — clicked at ${now}.`;
  });
}

