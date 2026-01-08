
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Cats vs Dogs Classifier")

# Load model (saved earlier )
model = tf.keras.models.load_model("cats_dogs_mobilenetv2.keras")

IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ["cat", "dog"]

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]

    return {
        "prediction": label,
        "confidence": float(prediction)
    }
