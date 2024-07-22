import spline_functions
import data_handler

# Data Variables
tickers = ["^GSPC", "^DJI", "^W5000"]
Reset_data = False
Full_taylor_degree = 10

# Model Variables
ticker_name = "SMP500"
data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Approximation_degree = 4
init_params = [1440.2223347981771,-0.3334441460503513,-0.31889987521700885,0.05700005425346717,0.011977403428826012]  # length must be one more than the approximation degree
num_of_nodes = 365*3 #3yr
Interval_length = 1

#Print node info?
Node_info= False

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
    if Node_info:
        Functions.Nodes_info(all=True)

    print('fin')
    print(f'Total length= {num_of_nodes*Interval_length} days')
