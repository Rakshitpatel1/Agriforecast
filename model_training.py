import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import joblib
import json
import requests
from datetime import datetime

# Configuration
DATA_DIR = "data/"
MODELS_DIR = "models/"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Get API key from environment variable
api_key = os.getenv("data.gov.in_api_key")

def load_or_fetch_data(resource_id, crop_name, region, start_date, end_date, refresh=False):
    """
    Load data from local storage or fetch from API if not available
    
    Parameters:
    resource_id (str): API resource ID
    crop_name (str): Name of the crop
    region (str): Region/market name
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    refresh (bool): Force refresh data from API
    
    Returns:
    pandas.DataFrame: Historical price data
    """
    # Create a unique filename for this data
    filename = f"{DATA_DIR}{crop_name}_{region}_{start_date}_{end_date}.csv"
    
    # Check if we already have this data locally
    if os.path.exists(filename) and not refresh:
        print(f"Loading data from local file: {filename}")
        return pd.read_csv(filename, parse_dates=["date"])
    
    # If not available locally or refresh requested, fetch from API
    print(f"Fetching data from API for {crop_name} in {region}...")
    
    try:
        # Try to fetch real data if API key is available
        if api_key:
            base_url = "https://api.data.gov.in/resource/"
            params = {
                "api-key": api_key,
                "format": "json",
                "filters[commodity]": crop_name,
                "filters[market]": region,
                "filters[from_date]": start_date,
                "filters[to_date]": end_date,
                "limit": 1000  # Adjust as needed
            }
            
            response = requests.get(f"{base_url}{resource_id}", params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "records" in data and len(data["records"]) > 0:
                    records = data["records"]
                    
                    # Process the response based on actual API structure
                    # Note: Actual field names might vary, adjust as needed
                    df = pd.DataFrame(records)
                    
                    # Rename columns to match our expected format
                    # Adjust column mappings based on actual API response
                    column_mapping = {
                        "arrival_date": "date",
                        "modal_price": "price",
                        "commodity": "crop",
                        "market": "region"
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Ensure required columns exist
                    if "date" not in df.columns or "price" not in df.columns:
                        raise ValueError("API response doesn't contain required fields")
                    
                    # Convert data types
                    df["date"] = pd.to_datetime(df["date"])
                    df["price"] = pd.to_numeric(df["price"])
                    
                    # Save to local storage for future use
                    df.to_csv(filename, index=False)
                    print(f"Data saved to {filename}")
                    
                    return df
    except Exception as e:
        print(f"Error fetching from API: {e}")
    
    # Generate mock data as fallback
    print("Using generated mock data (API fetch failed or no API key provided)")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(date_range)
    
    # Create a price trend with some seasonality and randomness
    base_price = 50 if crop_name in ["Rice", "Wheat"] else 20
    seasonal = 10 * np.sin(np.linspace(0, 6*np.pi, n))  # Seasonal component
    trend = np.linspace(0, 15, n)  # Upward trend
    random = np.random.normal(0, 5, n)  # Random noise
    
    prices = base_price + seasonal + trend + random
    prices = np.maximum(prices, 5)  # Ensure no negative prices
    
    df = pd.DataFrame({
        "date": date_range,
        "price": prices,
        "crop": crop_name,
        "region": region
    })
    
    # Save to local storage for future use
    df.to_csv(filename, index=False)
    print(f"Mock data saved to {filename}")
    
    return df

def engineer_features(df):
    """
    Create features for model training
    
    Parameters:
    df (pandas.DataFrame): Historical price data
    
    Returns:
    pandas.DataFrame: DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure date is datetime
    if df["date"].dtype != 'datetime64[ns]':
        df["date"] = pd.to_datetime(df["date"])
    
    # Add time-based features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    
    # Add lag features (previous prices)
    for lag in [1, 3, 7, 14, 30]:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        df[f"price_rolling_mean_{window}"] = df["price"].rolling(window=window).mean()
        df[f"price_rolling_std_{window}"] = df["price"].rolling(window=window).std()
        df[f"price_rolling_min_{window}"] = df["price"].rolling(window=window).min()
        df[f"price_rolling_max_{window}"] = df["price"].rolling(window=window).max()
    
    # Add price momentum features
    for window in [7, 14, 30]:
        df[f"price_momentum_{window}"] = df["price"] - df[f"price_rolling_mean_{window}"]
    
    # Drop rows with NaN values (from lag/rolling features)
    df = df.dropna()
    
    return df

def train_model(df, model_type="linear", test_size=0.2):
    """
    Train a price prediction model
    
    Parameters:
    df (pandas.DataFrame): DataFrame with engineered features
    model_type (str): Type of model to train ("linear" or "forest")
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: Trained model, feature list, model metrics
    """
    # Select features and target
    # Exclude date, crop and region columns from features
    feature_columns = [col for col in df.columns if col not in ["date", "crop", "region", "price"]]
    X = df[feature_columns]
    y = df["price"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model based on specified type
    if model_type == "forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # default to linear
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate feature importance for RandomForest
    if model_type == "forest":
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
    else:
        feature_importance = dict(zip(feature_columns, model.coef_))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [f[0] for f in sorted_features[:10]]
    
    # Package metrics
    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2,
        "top_features": top_features,
        "feature_importance": {k: float(v) for k, v in feature_importance.items()}
    }
    
    return model, feature_columns, metrics

def save_model(model, crop_name, region, feature_columns, metrics, model_type="linear"):
    """
    Save the trained model and its metadata
    
    Parameters:
    model: Trained model object
    crop_name (str): Name of the crop
    region (str): Region/market name
    feature_columns (list): List of feature column names
    metrics (dict): Model performance metrics
    model_type (str): Type of model ("linear" or "forest")
    
    Returns:
    str: Path to the saved model
    """
    # Create a unique model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{MODELS_DIR}{crop_name}_{region}_{model_type}_{timestamp}.joblib"
    
    # Save the model
    joblib.dump(model, model_filename)
    
    # Save metadata
    metadata = {
        "crop": crop_name,
        "region": region,
        "model_type": model_type,
        "features": feature_columns,
        "metrics": metrics,
        "created_at": timestamp
    }
    
    metadata_filename = f"{model_filename}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_filename}")
    print(f"Metadata saved to {metadata_filename}")
    
    return model_filename

def find_latest_model(crop_name, region, model_type="linear"):
    """
    Find the latest model for a specific crop and region
    
    Parameters:
    crop_name (str): Name of the crop
    region (str): Region/market name
    model_type (str): Type of model ("linear" or "forest")
    
    Returns:
    tuple: Path to the model file and its metadata
    """
    prefix = f"{crop_name}_{region}_{model_type}"
    
    # List all matching model files
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith(".joblib")]
    
    if not model_files:
        return None, None
    
    # Sort by timestamp (which is at the end of the filename)
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(MODELS_DIR, latest_model)
    
    # Load metadata
    metadata_path = f"{model_path}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
    
    return model_path, metadata

def train_all_models(crop_list, region_list, start_date, end_date, model_types=["linear", "forest"]):
    """
    Train and save models for all combinations of crops and regions
    
    Parameters:
    crop_list (list): List of crop names
    region_list (list): List of region names
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    model_types (list): List of model types to train
    
    Returns:
    dict: Dictionary of trained model paths
    """
    results = {}
    
    for crop in crop_list:
        results[crop] = {}
        
        for region in region_list:
            results[crop][region] = {}
            
            # Load or fetch data
            df = load_or_fetch_data("variety-wise-daily-market-prices-data-commodity", 
                                   crop, region, start_date, end_date)
            
            # Engineer features
            df_features = engineer_features(df)
            
            # Train models
            for model_type in model_types:
                print(f"Training {model_type} model for {crop} in {region}...")
                model, features, metrics = train_model(df_features, model_type)
                
                # Save model
                model_path = save_model(model, crop, region, features, metrics, model_type)
                
                # Store results
                results[crop][region][model_type] = {
                    "model_path": model_path,
                    "metrics": metrics
                }
                
                print(f"Model performance: MSE = {metrics['mse']:.2f}, R² = {metrics['r2']:.2f}")
                print("Top features:", metrics['top_features'][:5])
                print("-" * 50)
    
    return results

if __name__ == "__main__":
    # Define parameters
    crop_list = ["Rice", "Wheat", "Potato", "Onion", "Tomato"]
    region_list = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"]
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    
    # Train all models
    results = train_all_models(crop_list, region_list, start_date, end_date)
    
    # Print summary
    print("\nTraining Summary:")
    print("=" * 50)
    
    for crop in results:
        for region in results[crop]:
            for model_type in results[crop][region]:
                metrics = results[crop][region][model_type]["metrics"]
                print(f"{crop} in {region} ({model_type}): R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
    
    print("\nTraining complete! Models are saved in the 'models/' directory.")
