import pandas as pd
import numpy as np
import os

Degree_of_taylor = 10

# Specify the path to the folder
folder_path = "Stock_data/Raw_data"
processed_folder_path = "Stock_data/Processed_data"

# Ensure the folders exist
os.makedirs(folder_path, exist_ok=True)
os.makedirs(processed_folder_path, exist_ok=True)

# List all files in the folder
files = os.listdir(folder_path)

# Modifies each file, saves the modified file to the processed folder
for file in files:
    file_path = os.path.join(folder_path, file)
    processed_file_path = os.path.join(processed_folder_path, file)
    
    # Remove the original file if needed
    if os.path.exists(processed_file_path):
        print(f'Removing {processed_file_path}')
        os.remove(processed_file_path)
    
    # Only takes the opening value for each day
    df = pd.read_csv(file_path)[['Open']]
    df.rename(columns={'Open': 'Derivative_0'}, inplace=True)

    # Calculate N number of derivatives
    for i in range(Degree_of_taylor - 1):
        df['Derivative_' + str(i + 1)] = df['Derivative_0'].diff(i + 1)
    
    # Drop rows with NaN values (incomplete rows due to differencing)
    df.dropna(inplace=True)
    
    # Save the data to the specified CSV file in the processed folder
    df.to_csv(processed_file_path, index=False)
    
    print(f"Data has been processed and saved for {file}")
