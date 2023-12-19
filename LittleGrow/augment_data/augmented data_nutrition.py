import pandas as pd
import numpy as np

# Sample data
data = {
    'food_item': ['Apple', 'Banana', 'Chicken Breast', 'Spinach', 'Salmon', 'Brown Rice', 'Greek Yogurt', 'Almonds', 'Broccoli', 'Lentils'],
    'calories': [52, 105, 165, 23, 206, 215, 100, 7, 55, 230],
    'protein': [0.3, 1.3, 31.0, 2.9, 22.0, 5.0, 10.0, 0.3, 3.7, 18.0],
    'fat': [0.2, 0.3, 3.6, 0.4, 13.0, 1.6, 2.0, 0.6, 0.6, 1.0],
    'carbohydrates': [14, 27, 0, 3.6, 0, 45, 3, 0.6, 11.2, 40],
    'price': [1.0, 0.5, 3.0, 2.0, 5.0, 1.5, 1.8, 1.2, 2.5, 1.2],
    'label': ['normal', 'stunted', 'normal', 'stunted', 'normal', 'stunted', 'normal', 'stunted', 'normal', 'stunted']
}

# Function for data augmentation
def augment_data(df, num_samples=3000):
    augmented_data = []

    for _ in range(num_samples):
        # Randomly select a row from the original data
        original_row = df.iloc[np.random.randint(0, len(df))]

        # Create a new row with variations
        new_row = original_row.copy()

        # Random Scaling of Nutritional Values
        scaling_factor = np.random.uniform(0.8, 1.2)
        new_row[['calories', 'protein', 'fat', 'carbohydrates']] *= scaling_factor

        # Random Price Fluctuations
        new_row['price'] += np.random.uniform(0.1, 0.5)

        # Label Flipping
        new_row['label'] = np.random.choice(['normal', 'stunted'])

        augmented_data.append(new_row)

    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

# Augment the data with 1000 samples
augmented_df = augment_data(pd.DataFrame(data), num_samples=3000)

# Save the augmented data to a CSV file
augmented_df.to_csv('augmented_food_data.csv', index=False)

# Display the first few rows of the augmented data
print(augmented_df.head())
