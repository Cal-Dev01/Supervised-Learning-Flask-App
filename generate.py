import pandas as pd
import numpy as np


# Function to generate random animal data
def generate_animal_data(num_samples, animal_type):
    np.random.seed()
    size = np.random.uniform(0.5, 10.0, num_samples)
    sound = np.random.uniform(0.1, 5.0, num_samples)
    animal_type = [animal_type] * num_samples
    return pd.DataFrame({'size': size, 'sound': sound, 'type': animal_type})


# Generate sample data for cats, dogs, and new animals (e.g., rabbits)
cat_data = generate_animal_data(50, 'cat')
dog_data = generate_animal_data(50, 'dog')
rabbit_data = generate_animal_data(50, 'rabbit')

# Concatenate all data into one DataFrame
all_data = pd.concat([cat_data, dog_data, rabbit_data], ignore_index=True)

# Save the DataFrame to a CSV file
file_path = 'animal_data.csv'
all_data.to_csv(file_path, index=False)

print(f"CSV file generated and saved as {file_path}")
