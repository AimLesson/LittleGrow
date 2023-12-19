import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed augmented food dataset
csv_filename = 'preprocessed_augmented_food_dataset.csv'
food_df = pd.read_csv(csv_filename)

# Display the first few rows of the dataset
print("Preprocessed Augmented Food Dataset:")
print(food_df.head())

# Separate features (X) and target variable (y)
X = food_df.drop('price', axis=1)
y = food_df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
