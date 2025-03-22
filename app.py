from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import glob

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
    
    # Generate timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_filename = f"predictions_{zone}_{timestamp}.csv"
    latest_filename = f"predictions_{zone}_latest.csv"

    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=["latitude", "longitude"])
    predictions_df.to_csv(pred_filename, index=False)
    predictions_df.to_csv(latest_filename, index=False)  # Keep latest predictions easily accessible

    return {"message": f"✅ Predictions saved as {pred_filename}", "latest_file": latest_filename}

@app.post("/evaluate")
def evaluate(
    zone: str = Query(..., description="Zone to evaluate model performance on"),
    pred_file: str = Query(None, description="Prediction file to use (default is latest prediction)")
):
    val_file = f"seismic_{zone}_val.csv"
    default_pred_file = f"predictions_{zone}_latest.csv"
    pred_file = pred_file if pred_file else default_pred_file

    if not os.path.exists(val_file):
        raise HTTPException(status_code=404, detail=f"Validation dataset {val_file} not found!")

    if not os.path.exists(pred_file):
        raise HTTPException(status_code=404, detail=f"Predictions file {pred_file} not found! Run /predict first.")

    df = pd.read_csv(val_file)
    y_true = df[["latitude", "longitude"]]

    y_pred = pd.read_csv(pred_file)

    mse_lat = mean_squared_error(y_true["latitude"], y_pred["latitude"])
    mse_lon = mean_squared_error(y_true["longitude"], y_pred["longitude"])

    r2_lat = r2_score(y_true["latitude"], y_pred["latitude"])
    r2_lon = r2_score(y_true["longitude"], y_pred["longitude"])

    mae_lat = mean_absolute_error(y_true["latitude"], y_pred["latitude"])
    mae_lon = mean_absolute_error(y_true["longitude"], y_pred["longitude"])

    return {
        "MSE": {"latitude": mse_lat, "longitude": mse_lon},
        "R2": {"latitude": r2_lat, "longitude": r2_lon},
        "MAE": {"latitude": mae_lat, "longitude": mae_lon},
        "evaluated_predictions": pred_file
    }

@app.get("/list_predictions")
def list_predictions():
    pred_files = glob.glob("predictions_*.csv")
    latest_predictions = {}

    for file in pred_files:
        if "_latest.csv" in file:
            zone = file.split("_")[1] 
            latest_predictions[zone] = file

    return {
        "available_predictions": pred_files,
        "latest_per_zone": latest_predictions
    }

@app.post("/reload_model")
def reload_model(model_name: str):
    global model
    if not os.path.exists(model_name):
        return {"error": f"Model {model_name} not found!"}
    
    model = joblib.load(model_name)
    return {"message": f"✅ Model {model_name} successfully reloaded!"}

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
