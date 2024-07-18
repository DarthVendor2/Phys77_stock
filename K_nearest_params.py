import os
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.misc import derivative

class Spline_functions():
    
    def __init__(self, file, k=5, taylor_degree=3):
        # Ensure the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        # Create DataFrame from CSV file
        self.file = file
        self.df = pd.read_csv(file)

        # Store k value and Taylor degree
        self.k = k
        self.taylor_degree = taylor_degree + 1 

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

        if len(params) != self.taylor_degree:
            raise ValueError(f"Number of parameters must be {self.taylor_degree}")

        # Create Spline_node instance for specific params
        node = self.Spline_node(self.knn, params)

        # Store the results
        self.nodes.append(node)

        # Retrieve rows corresponding to k-nearest neighbors 
        rows = self.df.iloc[node.indices.flatten()] 

        # Calculate output parameters for the node
        node.calculate_output_params(rows)

        # Return the node
        return node


    def get_nodes(self):
        return np.array(self.nodes)

    class Spline_function():
        def __init__(self, knn, input_params):
            # Reshape input_params to ensure it's a 2D array with the correct shape
            self.N=len(input_params)
            self.input_params = np.array(input_params).reshape(1, -1)

            # Compute k-nearest neighbors
            self.distances, self.indices = knn.kneighbors(self.input_params)

            # Initialize output_params and taylor_params
            self.output_params = None
            self.taylor_params = None
        
        def calculate_output_params(self, rows):
            # Compute mean across columns (axis=0) to get average values for rest of values
            avg = np.mean(rows.iloc[:, self.N:], axis=0).values
            
            # Concatenate input_params and avg horizontally
            self.output_params = np.hstack((self.input_params.flatten(), avg))
            
            # Compute Taylor series parameters using the averages
            self.taylor_params = np.array([v / math.factorial(i) for i, v in enumerate(self.output_params)])

        def taylor_function(self, X):
            Y = np.zeros_like(X)
            for i, v in enumerate(self.taylor_params):
                Y += v * X**i
            self.Y= Y
            return Y
        
        #Needs to compute backwards derivatives at the endpoint
        def taylor_nth_derivative_endpoint(self):
            for i in range(self.N):
                pass



        def get_taylor_params(self):
            return self.taylor_params
        
        def get_output_params(self):
            return self.output_params

# Example usage
data_file_path = "Stock_data/Processed_data/Dow Jones Industrial Average_data.csv"
temp = Spline_functions(data_file_path, taylor_degree= 4)

# Example usage of finding k nearest neighbors
params = [1079.1300048828125,3.1800537109375,12.030029296875,9.449951171875,19.0699462890625]  # Example parameters
node = temp.find_k_nearest(params)

# Accessing results
print("Distances to nearest neighbors:", node.distances)
print("Indices of nearest neighbors:", node.indices)
print("Taylor series parameters:", node.get_taylor_params())
print("Output parameters:", node.get_output_params())

# Plotting Taylor series function
X = np.linspace(0, 1, 100)
Y = node.taylor_function(X)
plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Taylor Series Approximation')
plt.grid(True)
plt.show()
