from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import tensorflow as tf

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

prod_model = tf.keras.models.load_model(MODELS_DIR / "1.keras")
beta_model = tf.keras.models.load_model(MODELS_DIR / "2.keras")

class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, World!"

def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))
   return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    image = read_file_as_image(bytes)
    img_batch = np.expand_dims(image, 0)
    predictions = prod_model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        "class": predicted_class,
        "confidence": float(confidence)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
