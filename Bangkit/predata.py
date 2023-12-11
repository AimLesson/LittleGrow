import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('nutrition (2).csv', usecols=['calories','proteins','fat','carbohydrate'])

# Explore the Data
print(df.head())
print(df.info())
print(df.describe())

# Handle Missing Values
print(df.isnull().sum())
df = df.dropna()
df = df.fillna(df.mean())

# Remove Duplicates
df = df.drop_duplicates()

# Print the preprocessed data
print(df)
# Save the Preprocessed Data to a new CSV file
df.to_csv('preprocessed_dataset.csv', index=False)
