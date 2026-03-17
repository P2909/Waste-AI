const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const historyDiv = document.getElementById('history'); // ✅ Add history section in HTML

predictBtn.addEventListener('click', async () => {
    if (fileInput.files.length === 0) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.textContent = "Predicting...";

    try {
        const response = await fetch("http://127.0.0.1:5001/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (data.prediction) {
            resultDiv.textContent = "Prediction: " + data.prediction;

            // ✅ After prediction, refresh history
            loadHistory();
        } else if (data.error) {
            resultDiv.textContent = "Error: " + data.error;
        }
    } catch (err) {
        resultDiv.textContent = "Error connecting to server!";
        console.error(err);
    }
});

// 🔹 Function to load prediction history
async function loadHistory() {
    try {
        const response = await fetch("http://127.0.0.1:5001/history/1"); // hardcoded user_id=1 for now
        const data = await response.json();

        if (data.length === 0) {
            historyDiv.innerHTML = "<p>No history yet.</p>";
            return;
        }

        // ✅ Render history
        historyDiv.innerHTML = data.map(item => `
            <p>
                <strong>${item.filename}</strong> → ${item.prediction} 
                <small>(${item.timestamp})</small>
            </p>
        `).join("");
    } catch (err) {
        historyDiv.innerHTML = "<p>Error loading history!</p>";
        console.error(err);
    }
}

// ✅ Load history on page load
loadHistory();
