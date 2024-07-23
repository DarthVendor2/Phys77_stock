import spline_functions
import data_handler

# Model Variables
Ticker_name = "SMP500"
Data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Approximation_degree = 2 #Must be an int greater than 0. 0 is just the endpoint value (no derivatives) put into knn
Init_params = [1440.2223347981771,-0.3334441460503513,-0.31889987521700885]  # length must be one more than the approximation degree
Num_of_nodes = 365 #Adjustable. 365 is just one years worth of functions 
Interval_length = 10 #days. Better when smaller because taylor series is a closer approximation. 


# Data Variables
Tickers = ["^GSPC", "^DJI", "^W5000"] #Tickers can be adjusted
Reset_data = False #Just so you don't have to wait every time you want to create a plot
Full_taylor_degree = 3 #The Taylor degree for each spliced function
Moving_average = Interval_length*3 + 20 #Nums are adjustable; 30 is pretty good too

#Print node info?
Node_info= False
Show_legend= False

# Actual function
if __name__ == "__main__":
    #Just checks to make sure everything is ok
    if Full_taylor_degree <= Approximation_degree:
        raise "Approximation degree must be less than the full taylor degree"
    
    if type(Interval_length) is not int or Interval_length<1:
        raise "Interval length must be a  positive integer"
    
    if type(Num_of_nodes) is not int or Num_of_nodes<1:
        raise "Number of nodes must be a positive integer"

    if type(Approximation_degree) is not int or Approximation_degree<0:
        raise "Approximation degree must be a positive integer"

    if type(Full_taylor_degree) is not int or Full_taylor_degree<1:
        raise "Full taylor degree must be a positive integer"


    #Data management here
    if Reset_data:
        # Create an instance of data_handler
        temp = data_handler.Data_handler()
        # List of tickers to add and process
        for ticker in Tickers:
            temp.add_ticker(ticker)
        temp.process_tickers(Full_taylor_degree, rolling= Moving_average)
    
    # Initiates spline function
    Functions = spline_functions.Spline_functions(
        Data_file_path, 
        Ticker_name,
        interval_length=Interval_length,
        taylor_degree=Approximation_degree
    )

    # Create nodes
    nodes, _ = Functions.Create_node(Init_params, size= Num_of_nodes)
    
    # Graph the functions
    Functions.graph_functions(show_legend=Show_legend)
    
    # Print information about nodes
    if Node_info:
        Functions.Nodes_info(all=True)

    print(f'Total length= {Num_of_nodes*Interval_length/365:.3f} years')
    print('fin')
