import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math

class Spline_functions:
    def __init__(self, file: str, ticker_name: str, interval_length: int = 3, k: int = 5, taylor_degree: int = 3, splicing_index: int = 0):
        """
        Initialize Spline_functions with data from a CSV file and set up k-NN model.

        :param file: Path to the CSV file containing data.
        :param ticker_name: Name of the ticker for labeling.
        :param interval_length: Length of the interval for spline fitting.
        :param k: Number of nearest neighbors for k-NN.
        :param taylor_degree: Degree of the Taylor series approximation.
        :param splicing_index: Index for splicing the Taylor series.
        """
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        self.file = file
        self.df = pd.read_csv(file)
        self.ticker_name = ticker_name
        self.X = np.linspace(0, interval_length, 100)
        self.interval_length = interval_length
        self.k = k
        self.taylor_degree = taylor_degree + 1
        self.splicing_index = splicing_index
        self.last_node_num = 0
        self.nodes = []

        # Initialize k-NN
        X_data = self.df.iloc[:, :self.taylor_degree].values
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(X_data)

    def get_rand_params_from_data(self) -> tuple[np.ndarray, int]:
        """
        Get random parameters from the DataFrame for creating a node.

        :return: Tuple containing parameters and row number.
        """
        N = len(self.df)
        rand_row = np.random.randint(low=0, high=N)
        rand_data = self.df.iloc[rand_row, :].values

        return rand_data[:self.taylor_degree], rand_row

    def get_params_from_row_num(self, row: int) -> tuple[np.ndarray, int]:
        """
        Get parameters from a specific row number.

        :param row: Row number to retrieve parameters from.
        :return: Parameters from the specified row and the row number.
        """
        return self.df.iloc[row, :self.taylor_degree].values, row

    def Create_node(self, params: np.ndarray, size: int = 1) -> tuple[list, np.ndarray]:
        """
        Create spline nodes with specified parameters.

        :param params: Parameters for the node.
        :param size: Number of nodes to create.
        :return: List of created nodes and final parameters.
        """
        if len(params) != self.taylor_degree:
            raise ValueError(f"Number of parameters must be {self.taylor_degree}")

        if not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer")

        dx = self.X[1] - self.X[0]

        nodes_created = []
        for i in range(size):
            node = self.Spline_function(self.knn, params, self.last_node_num)
            self.last_node_num += 1
            self.nodes.append(node)
            nodes_created.append(node)

            rows = self.df.iloc[node.indices.flatten()]

            if not isinstance(node.indices, np.ndarray) or node.indices.ndim != 2:
                raise ValueError("Indices should be a 2D numpy array")

            node.calculate_output_params(rows, self.splicing_index)
            _, params = node.taylor_function(self.X, dx, set_Y=True)

        return nodes_created, params

    def get_nodes(self) -> np.ndarray:
        """
        Return an array of all the nodes.

        :return: Array of nodes.
        """
        return np.array(self.nodes)

    def graph_functions(self, show_legend: bool = False, start: int = 0, end: int = -1) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot all spline functions.

        :param show_legend: Whether to show the legend in the plot.
        :param start: Starting index for plotting.
        :param end: Ending index for plotting.
        :return: Figure and axes of the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for node in self.get_nodes()[start:end]:
            node.graph_function(ax, self.X)
            self.X += self.interval_length
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Taylor Series Approximation of {self.ticker_name}')
        ax.grid(True)
        if show_legend:
            ax.legend()
        return fig, ax

    def Nodes_info(self, Node_num: int = 0, all: bool = False, range: list[int] = [0, -1]):
        """
        Print information about nodes.

        :param Node_num: Node number to display information for.
        :param all: Whether to print information for all nodes.
        :param range: Range of nodes to print information for.
        """
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
        def __init__(self, knn: NearestNeighbors, input_params: np.ndarray, node_num: int):
            """
            Initialize a spline function node.

            :param knn: k-NN model instance.
            :param input_params: Input parameters for the spline function.
            :param node_num: Node number.
            """
            self.node_num = node_num
            self.N = len(input_params)
            self.input_params = np.array(input_params).reshape(1, -1)
            self.distances, self.indices = knn.kneighbors(self.input_params)
            self.output_params = None
            self.range = None
            self.domain = None
            self.fnth = None
            self.Y = None

        def calculate_output_params(self, rows: pd.DataFrame, splicing_index: int):
            """
            Calculate output parameters for the spline function.

            :param rows: DataFrame containing the rows of nearest neighbors.
            :param splicing_index: Index for splicing the Taylor series.
            """
            avg = np.mean(rows.iloc[:, splicing_index + 1:], axis=0).values
            self.output_params = np.hstack((self.input_params.flatten(), avg))

        def taylor_function(self, X: np.ndarray, dx: float, set_Y: bool = False) -> tuple[np.ndarray, np.ndarray]:
            """
            Compute the Taylor series function.

            :param X: Input array for the function.
            :param dx: Differential step size.
            :param set_Y: Whether to set Y values for the spline function.
            :return: Tuple containing function values and nth derivatives.
            """
            Y = np.zeros_like(X)
            for i, v in enumerate(self.output_params):
                Y += (v * X**i) / math.factorial(i)
            if set_Y:
                self.Y = Y
            return Y, self.nth_derivative_endpoint(dx)

        def nth_derivative_endpoint(self, dx: float) -> np.ndarray:
            """
            Compute the nth derivatives at the endpoint of Y.

            :param dx: Differential step size.
            :return: Array of nth derivatives.
            """
            f = pd.Series(self.Y[-self.N:])
            f_nth = [f.iloc[-1]]
            for _ in range(self.N - 1):
                f = f.diff() / dx
                f_nth.append(f.iloc[-1])
            self.fnth = np.array(f_nth)
            return self.fnth

        def graph_function(self, ax: plt.Axes, X: np.ndarray):
            """
            Plot the spline function.

            :param ax: Matplotlib axes to plot on.
            :param X: X values for the plot.
            """
            ax.plot(X, self.Y, label=f'Function {self.node_num}')
            self.domain = (X[0], X[-1])
            self.range = (self.Y[0], self.Y[-1])

        def print_node(self):
            """
            Print information about the spline node.
            """
            print(f'Node {self.node_num}')
            print("Domain:", self.domain)
            print("Range:", self.range)
            print("Output parameters:", self.output_params)
            print("Nth derivatives at endpoint:", self.fnth)
            print("Distances to nearest neighbors:", self.distances)
            print("Indices of nearest neighbors:", self.indices)

        def get_distances(self) -> np.ndarray:
            """
            Get distances to nearest neighbors.

            :return: Distances to nearest neighbors.
            """
            return self.distances

#Last was attempting to apply weights