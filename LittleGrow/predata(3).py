import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the augmented food dataset
csv_filename = 'augmented_food_dataset.csv'
food_df = pd.read_csv(csv_filename)

# Display the first few rows of the raw dataset
print("Raw Augmented Food Dataset:")
print(food_df.head())

# Handle missing values (if any)
food_df.dropna(inplace=True)

# Encode categorical variable 'food_item' using Label Encoding
label_encoder = LabelEncoder()
food_df['food_item'] = label_encoder.fit_transform(food_df['food_item'])

# Split the data into features (X) and target variable (y)
X = food_df.drop('price', axis=1)  # Features (excluding the 'price' column)
y = food_df['price']  # Target variable

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the first few rows of the preprocessed dataset
preprocessed_food_df = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_food_df['price'] = y.reset_index(drop=True)
print("\nPreprocessed Augmented Food Dataset:")
print(preprocessed_food_df.head())

# Save the preprocessed food dataset as a CSV file
preprocessed_csv_filename = 'preprocessed_augmented_food_dataset.csv'
preprocessed_food_df.to_csv(preprocessed_csv_filename, index=False)

print(f"\nPreprocessed food dataset saved as {preprocessed_csv_filename}")
