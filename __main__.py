import model
import data_handler as dh
from matplotlib.pyplot import show as plt_show

# Data files; automate later
Ticker= '^W5000'

# Model Variables
Approximation_degree = 2 # Must be an int greater than 0
KNN_Neighbors = 3 # Must be an int greater than 0
Num_of_nodes = 10  # Adjustable
Interval_length = 5  # days

# Parameter Selection
Use_rand_params = True
Use_row_num = False
Row_num = -1 #-1 = last point
Use_init_params = False
Init_params = [25.16, -.25]

# Data Variables
Reset_data = True #Must be true inorder for following to take effect

Full_taylor_degree = 5
Weights= None
#[10,2,1,1,1,1] #Length must be one greater than Full_taylor_degree

Moving_average = 10 #Int or None

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

#Main function
if __name__ == "__main__":
    # Validate inputs
    validate_inputs()

    try:
        Raw_data_file_path, Processed_data_file_path = dh.get_file_path(Ticker)
    except:
        Reset_data= True
    
    # Data management
    if Reset_data:
        dh.add_and_process_ticker(
            Ticker, 
            start_date= start_date, 
            end_date= end_date, 
            Full_taylor_degree= Full_taylor_degree,
            weights= Weights,
            rolling=Moving_average)

        Raw_data_file_path, Processed_data_file_path = dh.get_file_path(Ticker)

    KNN_Weights= dh.get_weights(Ticker)

    # Initialize spline functions
    Functions = model.Spline_functions(
        Processed_data_file_path,
        Ticker,
        KNN_Weights,
        interval_length=Interval_length,
        k=KNN_Neighbors,
        taylor_degree=Approximation_degree
    )

    # Create nodes and print initial parameters
    if Use_rand_params:
        params, row = Functions.get_rand_params_from_data()
        _, _ = Functions.Create_node(params, size=Num_of_nodes)
    elif Use_row_num:
        params, row= Functions.get_params_from_row_num(Row_num)
        # Create nodes if necessary, or adjust according to needs
        _, _ = Functions.Create_node(params, size=Num_of_nodes)
    else:
        _, _ = Functions.Create_node(Init_params, size=Num_of_nodes)
        params= Init_params
        row = None

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
    print('Initial params were:', params)
    print(f'Total length = {Num_of_nodes * Interval_length / 365:.3f} years')
    print('fin')

    plt_show()