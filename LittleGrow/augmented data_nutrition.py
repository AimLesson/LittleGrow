import pandas as pd
import numpy as np

# Sample food data with nutrition and price
food_data = {
    'food_item': ['Rice', 'Chicken', 'Vegetables', 'Fruits', 'Milk'],
    'calories': [200, 250, 50, 80, 120],
    'protein': [5, 30, 2, 1, 8],
    'fat': [1, 10, 0.5, 0.3, 5],
    'carbohydrates': [45, 0, 10, 20, 12],
    'price': [2, 5, 3, 4, 3]
}

# Convert the data to a DataFrame
food_df = pd.DataFrame(food_data)

# Number of augmented samples to generate
num_samples = 1000

# Generate augmented data for food
augmented_food_data = []

for _ in range(num_samples):
    augmented_sample = {
        'food_item': np.random.choice(food_df['food_item']),
        'calories': np.random.uniform(0.8, 1.2) * np.random.choice(food_df['calories']),
        'protein': np.random.uniform(0.8, 1.2) * np.random.choice(food_df['protein']),
        'fat': np.random.uniform(0.8, 1.2) * np.random.choice(food_df['fat']),
        'carbohydrates': np.random.uniform(0.8, 1.2) * np.random.choice(food_df['carbohydrates']),
        'price': np.random.uniform(0.8, 1.2) * np.random.choice(food_df['price'])
    }
    augmented_food_data.append(augmented_sample)

# Convert augmented data to a DataFrame
augmented_food_df = pd.DataFrame(augmented_food_data)

# Save the augmented food dataset as a CSV file
csv_filename = 'augmented_food_dataset.csv'
augmented_food_df.to_csv(csv_filename, index=False)

print(f"Augmented food dataset saved as {csv_filename}")
