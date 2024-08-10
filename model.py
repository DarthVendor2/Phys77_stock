import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math

class Spline_functions:
    def __init__(self, file: str, ticker_name: str, interval_length: int = 3, k: int = 5, dx: float = 0.1, taylor_degree: int = 2, weights= None, start_day: int = -1):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        #Processed file path
        self.file = file

        self.k = k
        #Taylor degree of splines
        self.taylor_degree = taylor_degree + 1
        #transform form from processed to func
        self.weights= weights

        if start_day < 0:
            days = np.array(pd.read_csv(file).index)
            start_day = days[start_day]
        self.start_day = start_day
        #Gets data
        self.df = pd.read_csv(file)[0: start_day+1]
        self.ticker_name = ticker_name

        if interval_length/dx > taylor_degree:
            dx = interval_length / (2*taylor_degree)
        #Sets range of domain for each splin
        self.X = np.arange(0, interval_length + dx, dx)
        self.interval_length = interval_length

        self.last_node_num = 0
        self.nodes = []

        # Initialize k-NN
        X_data = self.df.iloc[:, :self.taylor_degree].values

        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(X_data)


    def get_rand_params_from_data(self):
        #Return random data row
        N = len(self.df)
        rand_row = np.random.randint(low=0, high=N)
        rand_data = self.df.iloc[rand_row, :].values

        return rand_data[:self.taylor_degree], rand_row

    def get_params_from_row_num(self, row: int):
        #Return data row
        return self.df.iloc[row, :self.taylor_degree].values, row
        
    def Create_node(self, params, size: int = 1):
        #Used to create splines
        #Size is num of nodes
        #Params are data point
        if len(params) != self.taylor_degree:
            raise ValueError(f"Number of parameters must be {self.taylor_degree}")

        if not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer")

        nodes_created = []

        N= len(params)
        div_weights= self.weights[:N]
        
        dx = self.X[1] - self.X[0]

        for i in range(size):
            node = self.Spline_function(self.knn, params, self.last_node_num)
            self.last_node_num += 1
            self.nodes.append(node)
            nodes_created.append(node)

            rows = self.df.iloc[node.indices.flatten()]

            node.calculate_output_params(rows, self.weights)
            _, params = node.taylor_function(self.X, dx, set_Y=True)
            params = params / div_weights
            
        return nodes_created, params

    def get_nodes(self):
        return np.array(self.nodes)

    def graph_functions(self, show_legend: bool = False, nodes_starting_index: int = 0, nodes_ending_index: int = -1):
        fig, ax = plt.subplots(figsize=(10, 10))
        #for node in self.get_nodes()[nodes_starting_index:nodes_ending_index]:
        for num0, node in enumerate(self.get_nodes()):
            X = self.X + self.interval_length*num0 + self.start_day
            node.graph_function(ax, X)
        ax.set_xlabel('Days since start of January 1st, 2000')
        ax.set_ylabel('Value per share ($)')
        ax.set_title(f'Spline Model of {self.ticker_name}')
        ax.grid(True)
        if show_legend:
            ax.legend()
        return fig, ax
        
    def get_prediction_data(self, data_resolution: int = 1): #how to make faster?
        #returns array of predictions
        days = np.array([])
        prediction = np.array([])
        X_base = np.arange(0, self.interval_length, data_resolution)
        for node_num, node in enumerate(self.get_nodes()):
            prediction_part, _ = node.taylor_function(X_base,data_resolution)
            prediction = np.concatenate((prediction,prediction_part))
            node_domain = X_base + self.interval_length*node_num + self.start_day
            days = np.concatenate((days,node_domain))
        return days, prediction
        
    def Nodes_info(self, Node_num: int = 0, all: bool = False, range: list[int] = [0, -1]):
    #returns info about nodes
        nodes = self.get_nodes()
        N = len(nodes)
        if all:
            start, end = range
            start = max(start, 0)
            end = min(end, N)
            for node in nodes[start:end]:
                node.print_node()
        else:
            if Node_num <= 0 or Node_num > N:
                nodes[-1].print_node()
            else:
                nodes[Node_num - 1].print_node()

    class Spline_function:
        def __init__(self, knn: NearestNeighbors, input_params, node_num: int):
            #The actual splines in the big function
            self.node_num = node_num
            self.N = len(input_params)
            self.input_params = np.array(input_params).reshape(1,-1)
            self.distances, self.indices = knn.kneighbors(self.input_params)
            self.output_params = None
            self.range = None
            self.domain = None
            self.fnth = None
            self.Y = None

        def calculate_output_params(self, rows: pd.DataFrame, KNN_Weights):
            #calculate output parameters
            #averages outputs of KNN
            #Splices them together with first param
            avg = np.mean(rows.iloc[:,1:], axis=0).values
            first_param= [self.input_params.flatten()[0]]
            
            self.output_params = np.hstack((first_param, avg))*KNN_Weights

        def taylor_function(self, X, dx, set_Y: bool = False):
            #returns Taylor series function
            Y = np.zeros_like(X)
            for i, v in enumerate(self.output_params):
                Y = Y + (v * (X)**i) / math.factorial(i)
            if set_Y:
                self.Y = Y
            return Y, self.nth_derivative_endpoint(dx)

        def nth_derivative_endpoint(self, dx):
            #Backwards derivative
            f = pd.Series(self.Y[-self.N:])
            f_nth = [f.iloc[-1]]
            for _ in range(self.N - 1):
                f = f.diff()/dx
                f_nth.append(f.iloc[-1])
            self.fnth = np.array(f_nth)
            return self.fnth

        def graph_function(self, ax, X):
            #plots function 
            if self.node_num == 0:
                label = 'Taylor Function 1...'
            else:
                label = None
            ax.plot(X, self.Y, label=label)
            self.domain = (X[0], X[-1])
            self.range = (self.Y[0], self.Y[-1])

        def print_node(self):
            #General infor for debugging
            print(f'Node {self.node_num}')
            print("Domain:", self.domain)
            print("Range:", self.range)
            print("Output parameters:", self.output_params)
            print("Nth derivatives at endpoint:", self.fnth)
            print("Distances to nearest neighbors:", self.distances)
            print("Indices of nearest neighbors:", self.indices)
        
        
