from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

BASE_MODEL_PATH = "zone_0_model.pkl"
model = joblib.load(BASE_MODEL_PATH)

FEATURES = ["depth", "mag", "gap", "dmin", "rms"]

class EarthquakeData(BaseModel):
    depth: float
    mag: float
    gap: float
    dmin: float
    rms: float

class RetrainData(BaseModel):
    features: List[EarthquakeData]
    latitude: List[float]
    longitude: List[float]

@app.get("/list_models")
def list_models():
    models = [f for f in os.listdir() if f.endswith(".pkl")]
    return {"saved_models": models}

@app.post("/predict")
def predict(data: EarthquakeData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df[FEATURES]).tolist()
    return {"prediction": prediction}

@app.post("/reload_model")
def reload_model(model_name: str):
    global model
    if not os.path.exists(model_name):
        return {"error": f"Model {model_name} not found!"}
    
    model = joblib.load(model_name)
    return {"message": f"✅ Model {model_name} successfully reloaded!"}

@app.post("/retrain")
def retrain(data: RetrainData, save_as: str = "zone_0_model_updated.pkl"):
    model = joblib.load(BASE_MODEL_PATH)
    
    df = pd.DataFrame([x.dict() for x in data.features])
    y_train = pd.DataFrame({"latitude": data.latitude, "longitude": data.longitude})

    # Retrain model
    model.fit(df[FEATURES], y_train)

    # Save updated model with the specified filename
    joblib.dump(model, save_as)
    
    return {"message": f"Model retrained and saved as {save_as}."}

@app.get("/download_model")
def download_model(model_name: str):
    if not os.path.exists(model_name):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(model_name, filename=model_name, media_type="application/octet-stream")

@app.post("/upload_model")
def upload_model(file: UploadFile = File(...)):
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    return {"message": f"✅ Model {file.filename} uploaded successfully!"}