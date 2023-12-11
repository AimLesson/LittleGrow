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
    'arm_circumference': np.random.uniform(8, 12, num_samples)  # Example arm circumference range
}

# Combine stunted growth data with normal growth data
normal_data = {
    'height': np.random.uniform(*normal_growth_range, num_samples),
    'weight': np.random.uniform(3.0, 7.0, num_samples),  # Example weight range
    'head_circumference': np.random.uniform(35, 45, num_samples),  # Example head circumference range
    'arm_circumference': np.random.uniform(9, 14, num_samples)  # Example arm circumference range
}

# Create a DataFrame
stunted_df = pd.DataFrame(stunted_data)
normal_df = pd.DataFrame(normal_data)

# Add a label indicating stunted or normal growth
stunted_df['label'] = 'stunted'
normal_df['label'] = 'normal'

# Combine the datasets
combined_df = pd.concat([stunted_df, normal_df])

# Shuffle the dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save the dataset as a CSV file
csv_filename = 'stunted_growth_dataset.csv'
combined_df.to_csv(csv_filename, index=False)

print(f"Dataset saved as {csv_filename}")
