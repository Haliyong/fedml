from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load initial model
MODEL_PATH = "zone_0_model.pkl"
model = joblib.load(MODEL_PATH)

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

@app.post("/predict")
def predict(data: EarthquakeData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df[FEATURES]).tolist()
    return {"prediction": prediction}

@app.post("/retrain")
def retrain(data: RetrainData):
    global model
    df = pd.DataFrame([x.dict() for x in data.features])
    y_train = pd.DataFrame({"latitude": data.latitude, "longitude": data.longitude})

    # Retrain model
    model.fit(df[FEATURES], y_train)

    # Save updated model
    joblib.dump(model, "zone_0_model_updated.pkl")
    return {"message": "Model retrained and saved."}