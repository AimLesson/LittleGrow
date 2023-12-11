import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
csv_filename = 'stunted_growth_dataset.csv'
df = pd.read_csv(csv_filename)

# Handle missing values (if any)
df.dropna(inplace=True)

# Encode the 'label' column (stunted or normal) to numeric values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the data into features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features (height, weight, head circumference, arm circumference)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled can be used for training and testing machine learning models

# Optionally, you can save the preprocessed data
preprocessed_csv_filename = 'preprocessed_stunted_growth_dataset.csv'
preprocessed_df = pd.DataFrame(X_train_scaled, columns=X.columns)
preprocessed_df['label'] = y_train.reset_index(drop=True)
preprocessed_df.to_csv(preprocessed_csv_filename, index=False)

print(f"Preprocessed data saved as {preprocessed_csv_filename}")
