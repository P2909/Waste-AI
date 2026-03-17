// upload.js
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewContainer = document.getElementById("preview-container");
const resultContainer = document.getElementById("result-container");
const extraInfo = document.getElementById("extra-info");

const predictedClassSpan = document.getElementById("predicted-class");
const adviceSpan = document.getElementById("advice");

// Drag & Drop Highlight
dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.style.background = "#d4f5d1";
});

dropArea.addEventListener("dragleave", () => {
  dropArea.style.background = "#f9f9f9";
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    handleFile(fileInput.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
});

// Handle File Upload + Prediction
function handleFile(file) {
  if (!file) return;

  // Show preview
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    previewContainer.style.display = "block";
  };
  reader.readAsDataURL(file);

  // Send file to backend
  const formData = new FormData();
  formData.append("file", file);

  fetch("http://127.0.0.1:5001/predict", { // Update port if needed
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      console.log("Backend Response:", data);

      const prediction = data.prediction || data.class;

      if (prediction) {
        predictedClassSpan.textContent = prediction;

        let advice = "";
        switch (prediction.toLowerCase()) {
          case "organic":
            advice = "Dispose in a compost bin or green waste bin.";
            break;
          case "recyclable":
            advice = "Place in the recycling bin (plastic, paper, metals).";
            break;
          case "hazardous":
            advice = "Take to a hazardous waste center. Do not throw in bins.";
            break;
          case "general":
            advice = "Dispose in general waste bin (non-recyclable items).";
            break;
          default:
            advice = "Could not classify waste. Please try again.";
        }

        adviceSpan.textContent = advice;
        resultContainer.style.display = "block";
        extraInfo.style.display = "block";

        // Save history in localStorage
        let history = JSON.parse(localStorage.getItem("history")) || [];
        history.push({
          image: preview.src,
          prediction: prediction,
          advice: advice,
          time: new Date().toLocaleString()
        });
        localStorage.setItem("history", JSON.stringify(history));
      } else {
        alert("No prediction received from server.");
      }
    })
    .catch(err => {
      console.error("Error:", err);
      alert("⚠️ Error connecting to server. Is Flask running?");
    });
}
