import os
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors

# Define the data_file class
class data_file():
    
    def __init__(self, file, k=5, taylor_degree=3):
        # Ensure the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        # Create DataFrame from CSV file
        self.file = file
        self.df = pd.read_csv(file)

        # Store k value and Taylor degree
        self.k = k
        self.taylor_degree = taylor_degree

        # Initialize k-NN and nodes
        self.nodes = []
        self.init_k_nearest()
    
    def init_k_nearest(self):
        # Select the first 'taylor_degree' columns as features (X_data)
        X_data = self.df.iloc[:, :self.taylor_degree].values
        
        # Create instance knn, which is the model
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(X_data)

    def find_k_nearest(self, params):

        if len(params) > self.taylor_degree or len(params) < 1:
            raise ValueError(f"Number of parameters must be between 1 and {self.taylor_degree}")
                             
        # Create Spline_node instance for specific params
        node = self.Spline_node(self.knn, params)

        # Store the results
        self.nodes.append(node)

        # Retrieve rows corresponding to k-nearest neighbors indices
        rows = self.df.iloc[node.indices.flatten()]  # Ensure indices are flattened to 1D array

        # Calculate output parameters for the node
        node.calculate_output_params(rows)

        # Return the node
        return node


    def get_nodes(self):
        return np.array(self.nodes)

    #class for k-NN computation
    class Spline_node():
        def __init__(self, knn, input_params):
            # Reshape input_params to ensure it's a 2D array with the correct shape
            self.input_params = np.array(input_params).reshape(1, -1)

            # Compute k-nearest neighbors
            self.distances, self.indices = knn.kneighbors(self.input_params)

            # Initialize taylor_params
            self.taylor_params = None
        
        def calculate_output_params(self, rows):
            # Compute mean across collumns (axis=0) to get average values
            avg = np.mean(rows, axis=0)
            
            # Compute Taylor series parameters using the averages
            self.taylor_params = [v / math.factorial(i) for i, v in enumerate(avg)]
            
            # Store output_params as a NumPy array
            self.output_params = np.array(avg)
        
        def get_taylor_params(self):
            return self.taylor_params
        
        def get_output_params(self):
            return self.output_params

# Example usage
data_file_path = "Stock_data/Processed_data/dowjones_data.csv"
temp = data_file(data_file_path)

# Example usage of finding k nearest neighbors
params = [28901.80078125, 6.30078125, 32.791015625]  # Example parameters
node = temp.find_k_nearest(params)

# Accessing results
print("Distances to nearest neighbors:", node.distances)
print("Indices of nearest neighbors:", node.indices)
print("Taylor series parameters:", node.get_taylor_params())
print("Output parameters:", node.get_output_params())
