# demo_predict.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pathlib import Path

# ------------------------s
# Paths
# ------------------------
PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "models/waste_classifier.h5"
IMG_SIZE = (224, 224)

# ------------------------
# Load model
# ------------------------
print("📂 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ------------------------
# Predict function
# ------------------------
def predict(img_path, class_names):
    print(f"🖼️ Loading image: {img_path}")
    if not os.path.exists(img_path):
        print("❌ Image not found!")
        return None

    # Load & preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0) # batch dimension

    # Run prediction
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    return class_names[pred_idx]

# ------------------------
# Run demo prediction
# ------------------------
if __name__ == "__main__":
    # 👇 Change this path to any image from your dataset
    test_img = "/Users/muskan/projects/waste-ai/data/raw/delhi/recyclable/000003.jpg"

    # Rebuild class names from training
    train_dir = PROJECT / "data/processed/train"
    class_names = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])

    print("🚀 Running prediction...")
    result = predict(test_img, class_names)
    if result:
        print("✅ Prediction:", result)
