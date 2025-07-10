# filename: main.py
import redis
import pandas as pd
import xgboost as xgb
import numpy as np
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
MODEL_REDIS_KEY = 'trained_xgboost_model'

# --- MLflow Setup ---
mlflow.set_experiment("xgboost-redis-inference")

# --- FastAPI App ---
app = FastAPI(title="Car Price Prediction API", version="1.0")

# --- Redis Connection ---
try:
    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_conn.ping()
    print("Connected to Redis")
except redis.exceptions.ConnectionError as e:
    print("Redis connection failed:", e)
    raise

# --- Load model from Redis ---
def load_model_from_redis(redis_conn, model_key):
    print(f"Loading model from Redis key: {model_key}")
    model_bytes = redis_conn.get(model_key)
    if model_bytes is None:
        raise ValueError("Model not found in Redis.")
    
    bst = xgb.Booster()
    bst.load_model(bytearray(model_bytes))
    print("Model loaded from Redis.")
    return bst

try:
    model = load_model_from_redis(redis_conn, MODEL_REDIS_KEY)
    model_features = model.feature_names
except ValueError as e:
    raise RuntimeError(str(e))

# --- Pydantic Request Schema ---

class CarInput(BaseModel):
    model_year: int = Field(..., description="Car model year (e.g., 2015)")
    milage: int = Field(..., description="Total mileage of the car (in miles)")
    accident: int = Field(..., description="Number of past accidents (0 = none, 1 = at least one)")
    clean_title: int = Field(..., description="1 if the car has a clean title, 0 otherwise")
    engine_hp: float = Field(..., description="Engine horsepower (e.g., 150.0)")
    engine_displacement_L: float = Field(..., description="Engine displacement in liters (e.g., 2.0)")
    num_cylinders: int = Field(..., description="Number of cylinders (e.g., 4, 6, 8)")
    is_turbo_supercharged: int = Field(..., description="1 if turbocharged or supercharged, 0 otherwise")
    is_luxury: int = Field(..., description="1 if the car is a luxury brand, 0 otherwise")
    is_electric: int = Field(..., description="1 if electric vehicle, 0 otherwise")
    is_hybrid: int = Field(..., description="1 if hybrid vehicle, 0 otherwise")
    fuel_type: str = Field(..., description="Type of fuel (e.g., 'Gasoline', 'Diesel', 'Electric')")
    transmission: str = Field(..., description="Transmission type (e.g., 'Automatic', 'Manual')")
    engine_fuel_detail: str = Field(..., description="Detailed fuel info (e.g., 'Regular Unleaded', 'Premium')")

# --- Preprocessing ---
def preprocess_user_input(raw_input_dict, model_feature_list):
    user_df = pd.DataFrame([raw_input_dict])
    current_year = 2025
    if 'model_year' in user_df.columns:
        user_df['car_age'] = current_year - user_df['model_year']
        if 'model_year' not in model_feature_list and 'car_age' in model_feature_list:
            user_df = user_df.drop(columns=['model_year'])

    if 'milage' in user_df.columns and 'car_age' in user_df.columns and 'milage_per_year' in model_feature_list:
        user_df['milage_per_year'] = user_df['milage'] / (user_df['car_age'] + 1e-6)

    raw_fuel_type = raw_input_dict.get('fuel_type', 'Unknown').strip()
    for col in model_feature_list:
        if col.startswith('fuel_type_'):
            user_df[col] = int(col == f'fuel_type_{raw_fuel_type}')

    raw_transmission = raw_input_dict.get('transmission', 'Unknown').strip()
    for col in model_feature_list:
        if col.startswith('transmission_'):
            user_df[col] = int(col == f'transmission_{raw_transmission}')

    raw_engine_fuel_detail = raw_input_dict.get('engine_fuel_detail', 'Unknown').strip()
    for col in model_feature_list:
        if col.startswith('engine_fuel_detail_'):
            user_df[col] = int(col == f'engine_fuel_detail_{raw_engine_fuel_detail}')

    final_features_df = pd.DataFrame(columns=model_feature_list)
    final_features_df.loc[0] = 0
    for col in user_df.columns:
        if col in final_features_df.columns:
            final_features_df.at[0, col] = user_df.at[0, col]
    final_features_df = final_features_df[model_feature_list]

    return final_features_df

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(car: CarInput):
    input_dict = car.dict()
    try:
        input_df = preprocess_user_input(input_dict, model_features)
        dmatrix = xgb.DMatrix(input_df)
        prediction = model.predict(dmatrix)
        predicted_price = float(np.expm1(prediction[0]))

        with mlflow.start_run():
            mlflow.log_params(input_dict)
            mlflow.set_tag("model_source", "Redis")
            mlflow.set_tag("interface", "FastAPI")
            mlflow.log_metric("predicted_price", predicted_price)

        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

