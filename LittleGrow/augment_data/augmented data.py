import numpy as np
import pandas as pd

# Number of samples for the stunted dataset
num_samples = 1000

# Define ranges for normal growth and stunted growth
normal_growth_range = (55, 75)  # in centimeters
stunted_growth_range = (45, 55)  # in centimeters

# Generate synthetic data for stunted growth
stunted_data = {
    'height': np.random.uniform(*stunted_growth_range, num_samples),
    'weight': np.random.uniform(2.5, 5.0, num_samples),  # Example weight range
    'head_circumference': np.random.uniform(30, 40, num_samples),  # Example head circumference range
    'arm_circumference': np.random.uniform(8, 12, num_samples),  # Example arm circumference range
    'history_of_illness': np.random.binomial(1, 0.2, num_samples),  # Example: 20% chance of having a history of illness
    'birth_spacing': np.random.uniform(1, 5, num_samples),  # Example birth spacing range
    'calories': np.random.uniform(150, 300, num_samples),  # Example calorie range
    'protein': np.random.uniform(5, 20, num_samples),  # Example protein range
    'fat': np.random.uniform(2, 15, num_samples),  # Example fat range
    'carbohydrates': np.random.uniform(15, 40, num_samples),  # Example carbohydrate range
    'socioeconomic_status': np.random.choice(['high', 'medium', 'low'], num_samples),
    'access_to_healthcare': np.random.choice(['yes', 'no'], num_samples),
    'food_label': np.random.choice(['Nasi Goreng', 'Gado-Gado', 'Soto Ayam', 'Bubur Ayam', 'Pecel Lele', 'Nasi Padang', 'Bakso', 'Rendang', 'Nasi Uduk', 'Lontong'], num_samples)
}

# Combine stunted growth data with normal growth data
normal_data = {
    'height': np.random.uniform(*normal_growth_range, num_samples),
    'weight': np.random.uniform(3.0, 7.0, num_samples),  # Example weight range
    'head_circumference': np.random.uniform(35, 45, num_samples),  # Example head circumference range
    'arm_circumference': np.random.uniform(9, 14, num_samples),  # Example arm circumference range
    'history_of_illness': np.random.binomial(1, 0.1, num_samples),  # Example: 10% chance of having a history of illness
    'birth_spacing': np.random.uniform(1, 5, num_samples),  # Example birth spacing range
    'calories': np.random.uniform(150, 300, num_samples),  # Example calorie range
    'protein': np.random.uniform(5, 20, num_samples),  # Example protein range
    'fat': np.random.uniform(2, 15, num_samples),  # Example fat range
    'carbohydrates': np.random.uniform(15, 40, num_samples),  # Example carbohydrate range
    'socioeconomic_status': np.random.choice(['high', 'medium', 'low'], num_samples),
    'access_to_healthcare': np.random.choice(['yes', 'no'], num_samples),
    'food_label': np.random.choice(['Nasi Goreng', 'Gado-Gado', 'Soto Ayam', 'Bubur Ayam', 'Pecel Lele', 'Nasi Padang', 'Bakso', 'Rendang', 'Nasi Uduk', 'Lontong'], num_samples)
}

# Additional sample data
additional_data = {
    'height': [52, 71, 68, 70, 60],
    'weight': [4.75, 6.79, 5.64, 4.24, 4.04],
    'head_circumference': [35.66, 40.25, 38.5, 36.8, 35.1],
    'arm_circumference': [9.48, 12.34, 10.5, 9.8, 9.2],
    'history_of_illness': [0, 0, 1, 0, 0],
    'birth_spacing': [4.22, 2.41, 1.71, 1.85, 4.47],
    'calories': [200, 300, 250, 180, 220],
    'protein': [10, 15, 12, 8, 10],
    'fat': [5, 10, 8, 4, 6],
    'carbohydrates': [25, 30, 28, 20, 22],
    'socioeconomic_status': ['high', 'medium', 'low', 'medium', 'high'],
    'access_to_healthcare': ['yes', 'yes', 'no', 'yes', 'yes'],
    'food_label': ['Nasi Goreng', 'Gado-Gado', 'Soto Ayam', 'Bubur Ayam', 'Pecel Lele']
}

# Create DataFrames
stunted_df = pd.DataFrame(stunted_data)
normal_df = pd.DataFrame(normal_data)
additional_df = pd.DataFrame(additional_data)

# Add a label indicating stunted or normal growth
stunted_df['label'] = 'stunted'
normal_df['label'] = 'normal'

# Combine the datasets
combined_df = pd.concat([stunted_df, normal_df, additional_df])

# Shuffle the dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save the dataset as a CSV file
csv_filename = 'stunted_growth_dataset.csv'
combined_df.to_csv(csv_filename, index=False)

print(f"Dataset saved as {csv_filename}")
