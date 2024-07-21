import os
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class Spline_functions():
    
    def __init__(self, file, ticker_name, interval_length=3, k=5, taylor_degree=3):
        # Ensure the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        # Create DataFrame from CSV file
        self.file = file
        self.df = pd.read_csv(file)

        # Store k value and Taylor degree
        self.ticker_name = ticker_name
        self.X = np.linspace(0, interval_length, 200)
        self.dx = self.X[1] - self.X[0]
        self.interval_length = interval_length
        self.k = k
        self.taylor_degree = taylor_degree + 1

        # Initialize k-NN and nodes
        self.last_node_num = 0
        self.nodes = []
        self.init_k_nearest()

    

    def init_k_nearest(self):
        # Select the first 'taylor_degree' columns as features (X_data)
        X_data = self.df.iloc[:, :self.taylor_degree].values

        # Create instance knn, which is the model
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(X_data)

    def Create_node(self, params, size=1):
        if len(params) != self.taylor_degree:
            raise ValueError(f"Number of parameters must be {self.taylor_degree}")
        
        nodes_created = []
        for i in range(size):
            # Create Spline_node instance for specific params
            node = self.Spline_function(self.knn, params, self.last_node_num)
            self.last_node_num += 1

            # Store the results
            self.nodes.append(node)
            nodes_created.append(node)

            # Retrieve rows corresponding to k-nearest neighbors 
            rows = self.df.iloc[node.indices.flatten()] 

            # Calculate output parameters for the node
            node.calculate_output_params(rows)

            _, params = node.taylor_function(self.X, self.dx, set_Y=True)

        # Return the node
        return nodes_created

    def get_nodes(self):
        return np.array(self.nodes)
    
    def graph_functions(self):
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        X= self.X
        for node in self.get_nodes():
            node.graph_function(ax, self.X)
            X+= self.interval_length
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Taylor Series Approximation')
        ax.grid(True)
        ax.legend()
        plt.show()

    def Nodes_info(self, Node_num=0, all=False):
        nodes = self.get_nodes()
        if all:
            for node in nodes:
                node.print_node()
        else:
            nodes[Node_num - 1].print_node()

    class Spline_function():
        def __init__(self, knn, input_params, node_num):
            self.node_num = node_num
            # Reshape input_params to ensure it's a 2D array with the correct shape
            self.N = len(input_params)
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

        def taylor_function(self, X, dx, set_Y=False):
            Y = np.zeros_like(X)

            for i, v in enumerate(self.taylor_params):
                Y += v * X**i
            
            if set_Y:
                self.Y = Y
            
            return Y, self.nth_derivative_endpoint(dx)
        
        def nth_derivative_endpoint(self, dx):

            f = pd.Series(self.Y[-self.N:])
            f_nth = []

            for i in range(self.N):
                f_nth.append(f.iloc[-1])
                f = f.diff() / dx
            
            self.fnth = np.array(f_nth)

            return np.array(f_nth)
        
        def graph_function(self, ax, X):
            ax.plot(X, self.Y, label=f'Function {self.node_num}')
        
        def get_taylor_params(self):
            return self.taylor_params
        
        def get_output_params(self):
            return self.output_params
        
        def print_node(self):
            print(f'Node {self.node_num}')
            print("Distances to nearest neighbors:", self.distances)
            print("Indices of nearest neighbors:", self.indices)
            print("Taylor series parameters:", self.get_taylor_params())
            print("Output parameters:", self.get_output_params())
            print("Nth derivatives at endpoint:", self.fnth)


