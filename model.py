import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Optional
import random
from fastapi.middleware.cors import CORSMiddleware


# FastAPI app
app = FastAPI(title="Mood Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define input and output models
class FitbitData(BaseModel):
    calories: float
    steps: float
    distance: float
    lightly_active_minutes: float
    moderately_active_minutes: float
    very_active_minutes: float
    sedentary_minutes: float
    sleep_duration: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    minutesAsleep: Optional[float] = None
    minutesAwake: Optional[float] = None
    sleep_deep_ratio: Optional[float] = None
    sleep_light_ratio: Optional[float] = None
    sleep_rem_ratio: Optional[float] = None
    resting_hr: Optional[float] = None
    bpm: Optional[float] = None
    rmssd: Optional[float] = None
    nightly_temperature: Optional[float] = None
    daily_temperature_variation: Optional[float] = None

class MoodPrediction(BaseModel):
    dominant_mood: str
    mood_probabilities: Dict[str, float]
    input_features: Dict[str, float]

# Define the feature columns (must match the model training)
feature_columns = [
    'calories', 'steps', 'distance', 'lightly_active_minutes', 
    'moderately_active_minutes', 'very_active_minutes', 'sedentary_minutes',
    'sleep_duration', 'sleep_efficiency', 'minutesAsleep', 'minutesAwake',
    'sleep_deep_ratio', 'sleep_light_ratio', 'sleep_rem_ratio',
    'resting_hr', 'bpm', 'rmssd', 'nightly_temperature', 'daily_temperature_variation'
]

# Define the mood columns
mood_columns = ['ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED', 'SAD', 'TENSE/ANXIOUS', 'TIRED']

# Sample data for random mood predictions (when model is not available)
sample_data = [
    {"dominant_mood": "HAPPY", "message": "You're feeling happy today!"},
    {"dominant_mood": "ALERT", "message": "You're feeling alert and attentive."},
    {"dominant_mood": "NEUTRAL", "message": "You're feeling neutral today."},
    {"dominant_mood": "RESTED/RELAXED", "message": "You're feeling well-rested and relaxed."},
    {"dominant_mood": "SAD", "message": "You might be feeling a bit sad today."},
    {"dominant_mood": "TENSE/ANXIOUS", "message": "You seem a bit tense or anxious."},
    {"dominant_mood": "TIRED", "message": "You're feeling tired today."}
]

# Model and scaler objects
model = None
scaler = None

# Function to load model and scaler from file if they exist
def load_model():
    global model, scaler
    try:
        if os.path.exists("mood_prediction_model.pkl") and os.path.exists("feature_scaler.pkl"):
            with open("mood_prediction_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("feature_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Function to train and save model if it doesn't exist
def train_and_save_model(data_path="daily_fitbit_sema_df_unprocessed.csv"):
    global model, scaler
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Prepare the features and target
        X = data[feature_columns].copy()
        y = data[mood_columns].copy()
        
        # Fill missing values
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Filter out rows where all mood values are NaN
        valid_idx = ~y.isna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Convert any remaining NaN in y to 0
        y = y.fillna(0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Build and train the model
        base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model = MultiOutputClassifier(base_clf, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        with open("mood_prediction_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

# Function to get dominant mood
def get_dominant_mood(prediction, probabilities):
    if not any(prediction):
        # If no mood is predicted, use the highest probability
        mood_idx = np.argmax(probabilities)
        return mood_columns[mood_idx]
    
    # Otherwise, return the first mood that's predicted as 1
    for i, val in enumerate(prediction):
        if val == 1:
            return mood_columns[i]
    
    return "NEUTRAL"  # Default if somehow nothing is predicted

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Mood Prediction API is running"}

@app.post("/predict_mood", response_model=MoodPrediction)
def predict_mood(data: FitbitData):
    # Convert input data to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # Fill missing values with median values (these would ideally come from training data)
    # For simplicity, we'll use 0 for any missing values
    df = df.fillna(0)
    
    # Make sure we have all required features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Use only the features the model was trained on
    df = df[feature_columns]
    
    # Check if model is loaded, otherwise return random prediction
    if model is None or scaler is None:
        # If model not loaded, return a random mood
        random_mood = random.choice(sample_data)
        
        # Create dummy probabilities (all low except the chosen one)
        probs = {mood: 0.1 for mood in mood_columns}
        probs[random_mood["dominant_mood"]] = 0.9
        
        return MoodPrediction(
            dominant_mood=random_mood["dominant_mood"],
            mood_probabilities=probs,
            input_features=input_dict
        )
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)
    
    # Get probabilities for each mood
    mood_probs = {}
    for i, mood in enumerate(mood_columns):
        # Get probability of class 1 for this mood
        # Note: Random Forest predict_proba returns probability for each class
        mood_probs[mood] = float(probabilities[i][0][1])
    
    # Get dominant mood
    dominant_mood = get_dominant_mood(prediction, list(mood_probs.values()))
    
    return MoodPrediction(
        dominant_mood=dominant_mood,
        mood_probabilities=mood_probs,
        input_features=input_dict
    )

@app.get("/random_entry_prediction")
def random_entry_prediction():
    """Returns a prediction for a random entry - useful for demo purposes"""
    # Create a random Fitbit data entry
    random_data = {
        "calories": random.uniform(1500, 3000),
        "steps": random.uniform(3000, 15000),
        "distance": random.uniform(2000, 12000),
        "lightly_active_minutes": random.uniform(100, 300),
        "moderately_active_minutes": random.uniform(10, 60),
        "very_active_minutes": random.uniform(0, 120),
        "sedentary_minutes": random.uniform(300, 800),
        "sleep_duration": random.uniform(25000000, 40000000),
        "sleep_efficiency": random.uniform(85, 97),
        "minutesAsleep": random.uniform(400, 540),
        "minutesAwake": random.uniform(20, 100),
        "sleep_deep_ratio": random.uniform(0.8, 1.5),
        "sleep_light_ratio": random.uniform(0.8, 1.2),
        "sleep_rem_ratio": random.uniform(0.9, 1.7),
        "resting_hr": random.uniform(60, 75),
        "bpm": random.uniform(65, 85),
        "rmssd": random.uniform(80, 120),
        "nightly_temperature": random.uniform(33, 35),
        "daily_temperature_variation": random.uniform(-3, -1)
    }
    
    # Create a Fitbit data object
    fitbit_data = FitbitData(**random_data)
    
    # Get prediction
    return predict_mood(fitbit_data)

# Try to load model on startup
@app.on_event("startup")
async def startup_event():
    # Try to load the model
    if not load_model():
        # If loading fails, try to train a new one
        print("Model not found, attempting to train new model...")
        if os.path.exists("daily_fitbit_sema_df_unprocessed.csv"):
            if train_and_save_model():
                print("Model trained and saved successfully")
            else:
                print("Failed to train model, will use random predictions")
        else:
            print("Training data not found, will use random predictions")

# Run the API server
if __name__ == "__main__":
    uvicorn.run("mood_api:app", host="0.0.0.0", port=8000, reload=True)