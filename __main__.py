import model
import data_handler
from matplotlib.pyplot import show as plt_show

# Configuration
KNN_Weight = 1.0  # Must be a float greater than 0

# Data files
Ticker_name = "SMP500"
Processed_data_file_path = "Stock_data/Processed_data/S&P 500_data.csv"
Raw_data_file_path = "Stock_data/Raw_data/S&P 500_data.csv"

# Model Variables
Approximation_degree = 4  # Must be an int greater than 0
KNN_Neighbors = 5  # Must be an int greater than 0
Num_of_nodes = 365  # Adjustable
Interval_length = 10  # days

# Parameter Selection
Use_rand_params = True
Use_row_num = False
Row_num = 37
Use_init_params = False
Init_params = [1186.4951231421494, -3.3609459381738884, 0.06388467383625561, -0.011963181095696688, 0.008435414420346649]

# Data Variables
Reset_data = False
Tickers = ["^GSPC", "^DJI", "^W5000"]
Full_taylor_degree = 5
Moving_average = Interval_length * 3 + 20

# Info Flags
Node_info = False
Show_legend = False
Overlap_data = True

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

def reset_data():
    """Reset and process data using data_handler."""
    temp = data_handler.Data_handler()
    for ticker in Tickers:
        temp.add_ticker(ticker)
    temp.process_tickers(Full_taylor_degree, rolling=Moving_average)

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
        print(f"{row=}")
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
            row = 0
        ax = plotter.plot(ax, start=int(row), show_legend=Show_legend)
        print("Plotting")

    # Print node information
    if Node_info:
        Functions.Nodes_info(all=True)

    # Final output
    print(f'Total length = {Num_of_nodes * Interval_length / 365:.3f} years')
    print('fin')
    print("Figure:", fig)
    print("Axes:", ax)

    plt_show()
