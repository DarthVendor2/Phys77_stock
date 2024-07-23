import spline_functions
import data_handler

# Model Variables
ticker_name = "SMP500"
data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Approximation_degree = 2
init_params = [1440.2223347981771,-0.3334441460503513,-0.31889987521700885]  # length must be one more than the approximation degree
num_of_nodes = 365
Interval_length = 7 #days

# Data Variables
tickers = ["^GSPC", "^DJI", "^W5000"]
Reset_data = False
Full_taylor_degree = 3
Moving_average = Interval_length*2 + 20 #Nums are adjustable, just 30 is pretty good too

#Print node info?
Node_info= False

# Actual function
if __name__ == "__main__":
    #Just checks to make sure everything is ok
    if Full_taylor_degree <= Approximation_degree:
        raise "Approximation degree must be less than the full taylor degree"
    
    if type(Interval_length) is not int or Interval_length<1:
        raise "Interval length must be a  positive integer"
    
    if type(num_of_nodes) is not int or num_of_nodes<1:
        raise "Number of nodes must be a positive integer"

    if type(Approximation_degree) is not int or Approximation_degree<1:
        raise "Approximation degree must be a positive integer"

    if type(Full_taylor_degree) is not int or Full_taylor_degree<1:
        raise "Full taylor degree must be a positive integer"


    #Data management here
    if Reset_data:
        # Create an instance of data_handler
        temp = data_handler.Data_handler()
        # List of tickers to add and process
        for ticker in tickers:
            temp.add_ticker(ticker)
        temp.process_tickers(Full_taylor_degree, rolling= Moving_average)
    
    # Initiates spline function
    Functions = spline_functions.Spline_functions(
        data_file_path, 
        ticker_name,
        interval_length=Interval_length,
        taylor_degree=Approximation_degree
    )

    # Create nodes
    nodes = Functions.Create_node(init_params, size= num_of_nodes)
    
    # Graph the functions
    Functions.graph_functions()
    
    # Print information about nodes
    if Node_info:
        Functions.Nodes_info(all=True)

    print(f'Total length= {num_of_nodes*Interval_length/365} years')
    print('fin')
