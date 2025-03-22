from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
def predict(zone: str = Query(..., description="Zone to load local validation data from")):
    file_path = f"seismic_{zone}_val.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset {file_path} not found!")

    df = pd.read_csv(file_path)
    X_test = df[FEATURES]

    predictions = model.predict(X_test).tolist()
    return {"predictions": predictions}

@app.post("/evaluate")
def evaluate(zone: str = Query(..., description="Zone to evaluate model performance on")):
    val_file = f"seismic_{zone}_val.csv"

    if not os.path.exists(val_file):
        raise HTTPException(status_code=404, detail=f"Validation dataset {val_file} not found!")

    df = pd.read_csv(val_file)
    X_test = df[FEATURES]
    y_true = df[["latitude", "longitude"]]

    predictions = model.predict(X_test)
    y_pred = pd.DataFrame(predictions, columns=["latitude", "longitude"])

    # Compute metrics
    mse_lat = mean_squared_error(y_true["latitude"], y_pred["latitude"])
    mse_lon = mean_squared_error(y_true["longitude"], y_pred["longitude"])

    r2_lat = r2_score(y_true["latitude"], y_pred["latitude"])
    r2_lon = r2_score(y_true["longitude"], y_pred["longitude"])

    mae_lat = mean_absolute_error(y_true["latitude"], y_pred["latitude"])
    mae_lon = mean_absolute_error(y_true["longitude"], y_pred["longitude"])

    return {
        "MSE": {"latitude": mse_lat, "longitude": mse_lon},
        "R2": {"latitude": r2_lat, "longitude": r2_lon},
        "MAE": {"latitude": mae_lat, "longitude": mae_lon}
    }

@app.post("/reload_model")
def reload_model(model_name: str):
    global model
    if not os.path.exists(model_name):
        return {"error": f"Model {model_name} not found!"}
    
    model = joblib.load(model_name)
    return {"message": f"✅ Model {model_name} successfully reloaded!"}

from fastapi import Query

@app.post("/retrain")
def retrain(zone: str = Query(..., description="Zone dataset to use for retraining"),
            save_as: str = "zone_0_model_updated.pkl"):
    file_path = f"seismic_{zone}.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset {file_path} not found!")

    df = pd.read_csv(file_path)
    X_train = df[FEATURES]
    y_train = df[["latitude", "longitude"]]

    # Retrain the model
    model.fit(X_train, y_train)

    # Save the updated model
    joblib.dump(model, save_as)

    return {"message": f"✅ Model retrained on {file_path} and saved as {save_as}."}

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

@app.get("/current_model")
def current_model():
    return {
        "message": "📢 Currently loaded model",
        "model_name": BASE_MODEL_PATH,
        "model_type": type(model).__name__
    }