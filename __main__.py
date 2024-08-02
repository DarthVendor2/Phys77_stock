import model
import data_handler
from matplotlib.pyplot import show as plt_show

# Add functionality later
KNN_Weight = 1.0  # Must be a float greater than 0
Add_noise= False

# Data files; automate later
Ticker_name = "^smp500"
Processed_data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Raw_data_file_path = "Stock_data/Raw_data/Dow Jones Industrial Average_data.csv"

# Model Variables
Approximation_degree = 1 # Must be an int greater than 0
Splicing_index=0 #Can be set
KNN_Neighbors = 10 # Must be an int greater than 0
Num_of_nodes = 20  # Adjustable
Interval_length = 3  # days

# Parameter Selection
Use_rand_params = True
Use_row_num = False
Row_num = -1 #-1 = last point
Use_init_params = False
Init_params = [25.16, -.25]

# Data Variables
Reset_data = True #Must be true inorder for following to take effect
Tickers = ["^GSPC", "^DJI", "^W5000"]
Full_taylor_degree = 5
Moving_average = 30
start_date="2000-01-01" 
end_date="2023-12-31"

# Info Flags
Node_info = True
Show_legend = False
Overlap_data = False

def validate_inputs():
    """Validate input parameters."""
    if sum([Use_rand_params, Use_row_num, Use_init_params]) != 1:
        raise ValueError("Set exactly one of Use_rand_params, Use_row_num, or Use_init_params to True.")

    if Full_taylor_degree <= Approximation_degree:
        raise ValueError("Approximation degree must be less than the full Taylor degree.")

    if not isinstance(Interval_length, int) or Interval_length < 1:
        raise ValueError("Interval length must be a positive integer.")

    if not isinstance(Num_of_nodes, int) or Num_of_nodes < 1:
        raise ValueError("Number of nodes must be a positive integer.")

    if not isinstance(Approximation_degree, int) or Approximation_degree < 0:
        raise ValueError("Approximation degree must be a non-negative integer.")

    if not isinstance(Full_taylor_degree, int) or Full_taylor_degree < 1:
        raise ValueError("Full Taylor degree must be a positive integer.")

    if not isinstance(Splicing_index, int) or Splicing_index < 0 or Splicing_index > Approximation_degree:
        raise ValueError("Splicing index must be a positive integer less than the approximation degree.")

def reset_data():
    """Reset and process data using data_handler."""
    temp = data_handler.Data_handler()
    for ticker in Tickers:
        temp.add_ticker(ticker, start_date=start_date, end_date=end_date)
    temp.process_tickers(Full_taylor_degree, rolling=Moving_average)

#Main function
if __name__ == "__main__":
    # Validate inputs
    validate_inputs()

    # Data management
    if Reset_data:
        reset_data()

    # Initialize spline functions
    Functions = model.Spline_functions(
        Processed_data_file_path,
        Ticker_name,
        interval_length=Interval_length,
        k=KNN_Neighbors,
        taylor_degree=Approximation_degree
    )

    # Create nodes and print initial parameters
    if Use_rand_params:
        params, row = Functions.get_rand_params_from_data()
        _, _ = Functions.Create_node(params, size=Num_of_nodes)
        print('Initial params were:', params)
    elif Use_row_num:
        params, row= Functions.get_params_from_row_num(Row_num)
        print('Initial params were:', params)
        # Create nodes if necessary, or adjust according to needs
        _, _ = Functions.Create_node(params, size=Num_of_nodes)
    else:
        _, _ = Functions.Create_node(Init_params, size=Num_of_nodes)
        row = None
        print('Initial params were:', Init_params)

    # Graph the functions
    fig, ax = Functions.graph_functions(show_legend=Show_legend)

    if Overlap_data and not Use_init_params:
        plotter = data_handler.Plotter(Raw_data_file_path)
        # Ensure row is an integer or handle None case in the plot method
        if row is None:
            pass
        else:
            ax = plotter.plot(ax, start=int(row),end= row + Num_of_nodes * Interval_length, show_legend=Show_legend)

    # Print node information
    if Node_info:
        Functions.Nodes_info(all=True)

    # Final output
    print(f'Total length = {Num_of_nodes * Interval_length / 365:.3f} years')
    print('fin')

    plt_show()
