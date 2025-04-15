
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import json
from typing import List, Optional
import os
from model_training import load_or_fetch_data, engineer_features, train_model, find_latest_model
import joblib
from datetime import datetime, timedelta

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    state: str
    market_yard: str
    crop: str
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    historical_prices: List[dict]
    predicted_prices: List[dict]
    metrics: dict
    model_info: dict

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    try:
        # Load historical data
        df = load_or_fetch_data(
            "variety-wise-daily-market-prices-data-commodity",  # Using the actual resource ID
            request.crop,
            request.market_yard, 
            request.start_date, 
            request.end_date
        )
        
        # Engineer features for model training
        df_features = engineer_features(df)
        
        # Train a model or load an existing one
        model_type = "forest"  # Using Random Forest for better predictions
        model, features, metrics = train_model(df_features, model_type=model_type)
        
        # Generate future dates for prediction (30 days from end_date)
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        future_start = end_date + timedelta(days=1)
        future_end = future_start + timedelta(days=30)
        
        # Generate predictions for future dates
        future_dates = pd.date_range(start=future_start, end=future_end)
        X_future = pd.DataFrame(index=future_dates)
        X_future["month"] = X_future.index.month
        X_future["year"] = X_future.index.year
        X_future["day_of_year"] = X_future.index.dayofyear
        X_future["day_of_month"] = X_future.index.day
        X_future["day_of_week"] = X_future.index.dayofweek
        X_future["week_of_year"] = X_future.index.isocalendar().week
        X_future["quarter"] = X_future.index.quarter
        
        # Use last known values for lag features
        last_prices = df["price"].iloc[-30:].to_list()
        if len(last_prices) < 30:
            # Pad with the first value if not enough data
            last_prices = [last_prices[0]] * (30 - len(last_prices)) + last_prices
            
        X_future["price_lag_1"] = [last_prices[-1]] * len(X_future)
        X_future["price_lag_3"] = [last_prices[-3]] * len(X_future)
        X_future["price_lag_7"] = [last_prices[-7]] * len(X_future)
        X_future["price_lag_14"] = [last_prices[-14]] * len(X_future)
        X_future["price_lag_30"] = [last_prices[-30]] * len(X_future)
        
        # Use last values for rolling features
        X_future["price_rolling_mean_7"] = df["price"].rolling(window=7).mean().iloc[-1]
        X_future["price_rolling_std_7"] = df["price"].rolling(window=7).std().iloc[-1]
        X_future["price_rolling_min_7"] = df["price"].rolling(window=7).min().iloc[-1]
        X_future["price_rolling_max_7"] = df["price"].rolling(window=7).max().iloc[-1]
        X_future["price_rolling_mean_14"] = df["price"].rolling(window=14).mean().iloc[-1]
        X_future["price_rolling_std_14"] = df["price"].rolling(window=14).std().iloc[-1]
        X_future["price_rolling_min_14"] = df["price"].rolling(window=14).min().iloc[-1]
        X_future["price_rolling_max_14"] = df["price"].rolling(window=14).max().iloc[-1]
        X_future["price_rolling_mean_30"] = df["price"].rolling(window=30).mean().iloc[-1]
        X_future["price_rolling_std_30"] = df["price"].rolling(window=30).std().iloc[-1]
        X_future["price_rolling_min_30"] = df["price"].rolling(window=30).min().iloc[-1]
        X_future["price_rolling_max_30"] = df["price"].rolling(window=30).max().iloc[-1]
        
        # Compute momentum features
        X_future["price_momentum_7"] = 0  # Simplified
        X_future["price_momentum_14"] = 0  # Simplified
        X_future["price_momentum_30"] = 0  # Simplified
        
        # Ensure all needed columns are present
        X_future = X_future[features]
        
        # Make predictions
        future_prices = model.predict(X_future)
        
        # Prepare response data
        historical_prices = [
            {"date": row["date"].strftime("%Y-%m-%d"), "price": float(row["price"])}
            for _, row in df.iterrows()
        ]
        
        predicted_prices = [
            {"date": date.strftime("%Y-%m-%d"), "price": float(price)}
            for date, price in zip(future_dates, future_prices)
        ]
        
        # Calculate trend information
        last_known_price = df["price"].iloc[-1]
        avg_predicted_price = float(np.mean(future_prices))
        predicted_trend = "rising" if avg_predicted_price > last_known_price else "falling"
        trend_percentage = abs((avg_predicted_price / last_known_price - 1) * 100)
        
        return {
            "historical_prices": historical_prices,
            "predicted_prices": predicted_prices,
            "metrics": {
                "mse": float(metrics["mse"]),
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "last_known_price": float(last_known_price),
                "avg_predicted_price": avg_predicted_price,
                "trend": predicted_trend,
                "trend_percentage": float(trend_percentage)
            },
            "model_info": {
                "model_type": "Random Forest" if model_type == "forest" else "Linear Regression",
                "top_features": metrics["top_features"][:5]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
