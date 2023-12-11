from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the preprocessed dataset
preprocessed_csv_filename = 'dataset/preprocessed_stunted_growth_dataset.csv'
df = pd.read_csv(preprocessed_csv_filename)

# Split the data into features (X) and labels (y)
X = df.drop('condition', axis=1)
y = df['condition']

# Scale numeric features (optional, depending on the algorithm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_scaled, y)

class Item(BaseModel):
    features: list

@app.post("/predict")
async def predict(item: Item):
    try:
        # Get input data from request
        features = item.features

        # Preprocess the input data
        input_data = pd.DataFrame([features])
        input_data_scaled = scaler.transform(input_data)

        # Make predictions
        prediction = classifier.predict(input_data_scaled)

        # Convert the NumPy result to a regular Python type
        prediction_result = prediction.item()

        # Return the prediction as JSON
        result = {'prediction': prediction_result}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
