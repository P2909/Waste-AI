from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
import os
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Database connection
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",              # apna MySQL username
        password="Muskan@1707",      # apna MySQL password
        database="waste_ai"       # jo DB tumne banayi
    )

# ✅ ML Model Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.pth")

print("🔍 Loading model from:", MODEL_PATH)

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

class_names = ["general", "hazardous", "organic", "recyclable"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- ML PREDICTION ----------
from datetime import datetime
import mysql.connector

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # 🔹 Image load + prediction
        img = Image.open(file.stream).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        # 🔹 Save prediction in DB
        db = get_db()
        cursor = db.cursor()

        # filhal user_id hardcode kar dete hain (login se pass karna hoga later)
        user_id = 1  
        cursor.execute(
            "INSERT INTO predictions (user_id, filename, prediction, timestamp) VALUES (%s, %s, %s, %s)",
            (user_id, file.filename, predicted_class, datetime.now())
        )
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- AUTH & DATABASE ROUTES ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username, password = data["username"], data["password"]

    db = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cur.fetchone():
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(password)
    cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
    db.commit()
    cur.close(); db.close()

    return jsonify({"message": "User registered"})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username, password = data["username"], data["password"]

    db = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cur.fetchone()
    cur.close(); db.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful", "user_id": user["id"]})

@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    data = request.json
    user_id, filename, prediction = data["user_id"], data["filename"], data["prediction"]

    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO predictions (user_id, filename, prediction) VALUES (%s, %s, %s)",
                (user_id, filename, prediction))
    db.commit()
    cur.close(); db.close()

    return jsonify({"message": "Prediction saved"})

@app.route("/history/<int:user_id>")
def history(user_id):
    db = get_db()
    cur = db.cursor(dictionary=True)
    
    # 🔹 ab timestamp bhi fetch karenge
    cur.execute("SELECT id, filename, prediction, timestamp FROM predictions WHERE user_id=%s", (user_id,))
    rows = cur.fetchall()
    cur.close()
    db.close()

    # 🔹 MySQL timestamp ko JSON-friendly string banayenge
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "filename": row["filename"],
            "prediction": row["prediction"],
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else None
        })

    return jsonify(history)


# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)