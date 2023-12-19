from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Load the dataset
df = pd.read_csv('stunted_growth_dataset(food).csv')

# Select relevant features and target variable
X = df[['height', 'weight', 'head_circumference', 'arm_circumference', 'history_of_illness', 'birth_spacing']]
y = df['food_label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KNN model with the best hyperparameters
best_k = 3  # Use the best_k value from your GridSearchCV
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_scaled, y)

app = FastAPI()

class InputData(BaseModel):
    height: float
    weight: float
    head_circumference: float
    arm_circumference: float
    history_of_illness: int
    birth_spacing: int

@app.post("/predictfood")
def predict(data: InputData):
    input_data = [[
        data.height,
        data.weight,
        data.head_circumference,
        data.arm_circumference,
        data.history_of_illness,
        data.birth_spacing
    ]]
    input_data_scaled = scaler.transform(input_data)
    prediction_proba = knn_model.predict_proba(input_data_scaled)[0]

    # Get the top predicted classes and their probabilities
    top_predictions = [
        {"label": label, "probability": float(proba)}
        for label, proba in zip(knn_model.classes_, prediction_proba)
    ]
    top_predictions.sort(key=lambda x: x["probability"], reverse=True)
    top_4_predictions = top_predictions[:4]

    return {"predictions": top_4_predictions}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
