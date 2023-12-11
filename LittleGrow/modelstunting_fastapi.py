from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI()

# Load the preprocessed newborn dataset for stunted and growth models
csv_filename = 'preprocessed_stunted_growth_dataset.csv'

# Load the combined newborn dataset
newborn_df = pd.read_csv(csv_filename)

# Check if 'condition' column is present
if 'condition' not in newborn_df.columns:
    raise ValueError("The 'condition' column is not present in the dataset.")

# Separate features (X) and target variable (y) for stunted and growth models
X_newborn = newborn_df.drop('condition', axis=1)
y_newborn = newborn_df['condition']

# Split the data into training and testing sets
X_train_newborn, X_test_newborn, y_train_newborn, y_test_newborn = train_test_split(
    X_newborn, y_newborn, test_size=0.2, random_state=42
)

# Train Gradient Boosting Classifier models for stunted and growth
model_stunted_growth = GradientBoostingClassifier(random_state=42)
model_stunted_growth.fit(X_train_newborn, y_train_newborn)

# Create a scaler for newborn data
scaler_newborn = StandardScaler()
scaler_newborn.fit(X_train_newborn)

# Create a Pydantic model for newborn input data validation
class NewbornCondition(BaseModel):
    height: float
    weight: float
    head_circumference: float
    arm_circumference: float

# API endpoint for predicting stunted or growth condition
@app.post("/predict_condition")
def predict_condition(newborn: NewbornCondition):
    input_data = [
        newborn.height,
        newborn.weight,
        newborn.head_circumference,
        newborn.arm_circumference
    ]

    # Scale input data using the same scaler used during preprocessing
    input_data_scaled = scaler_newborn.transform([input_data])

    # Make predictions using the trained model
    prediction = model_stunted_growth.predict(input_data_scaled)[0]

    return {"predicted_condition": "stunted" if prediction == 1 else "growth" if prediction == 2 else "normal"}
