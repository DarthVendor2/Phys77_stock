import spline_functions
import data_handler
import numpy as np

# Data Variables
tickers = ["^GSPC", "^DJI", "^W5000"]
Reset_data = True
Full_taylor_degree = 9

# Model Variables
ticker_name = "SMP500"
data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Approximation_degree = 3
init_params = [1448.719970703125,-1.09366455078125,-0.11256686740451387,-0.0009277795862268512]  # length must be one more than the approximation degree
num_of_nodes = 100
Interval_length = 3

# Actual function
if __name__ == "__main__":
    
    if Reset_data:
        # Create an instance of data_handler
        temp = data_handler.data_handler()
        # List of tickers to add and process
        for ticker in tickers:
            temp.add_ticker(ticker)
        temp.process_tickers(Full_taylor_degree)
    
    # Create an instance of Spline_functions
    Functions = spline_functions.Spline_functions(
        data_file_path, 
        ticker_name,
        interval_length=Interval_length,
        taylor_degree=Approximation_degree
    )

    # Create nodes
    nodes = Functions.Create_node(init_params, size=num_of_nodes)
    
    # Graph the functions
    Functions.graph_functions()
    
    # Print information about nodes
    Functions.Nodes_info(all=True)

    print('fin')
