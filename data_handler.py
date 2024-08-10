import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def add_ticker(ticker_label, start_date, end_date):
    #add raw data csv
    #updates record
    Raw_data_folder_path = "Stock_data/Raw_data"

    try:
        # Get ticker information
        ticker = yf.Ticker(ticker_label)
        company_name = ticker.info.get('longName', ticker_label)
        file_path = os.path.join(Raw_data_folder_path, f'Raw_{company_name}_data.csv')

        # Download historical data
        df = yf.download(ticker_label, start=start_date, end=end_date)
        df.to_csv(file_path, index=False)

        # Append to record
        append_to_record(ticker_label, company_name, "Raw", file_path)

        return df, company_name

    except Exception as e:
        raise Exception(f"Error adding ticker {ticker_label}: {e}")

def process_ticker(ticker, company_name, df, process_taylor_degree, weights = None, rolling = None):
    #processes data
    Processed_data_folder_path = "Stock_data/Processed_data"
    file_path = os.path.join(Processed_data_folder_path, f'Processed_{company_name}_data.csv')

    try:
        # Extract 'Open' prices and rename column
        df = df["Open"].to_frame()
        df.rename(columns={'Open': 'Derivative_0'}, inplace=True)

        # Calculate rolling derivatives
        df = rolling_derivatives(df, process_taylor_degree, rolling)


        # Calculate KNN Weights
        if weights is None:
            KNN_Weight = np.ones(process_taylor_degree + 1)
        else:
            KNN_Weight = np.abs(np.array(df.iloc[0])) / np.array(weights)
            df = df / KNN_Weight

        # Save processed data
        df.to_csv(file_path, index=False)

        # Append to record
        append_to_record(ticker, company_name, "Processed", file_path)

        Update_weight(ticker, KNN_Weight)

        return KNN_Weight

    except Exception as e:
        raise Exception(f"Error processing ticker {ticker}: {e}")

def add_and_process_ticker(ticker_label, start_date="2000-01-01", end_date="2023-12-31", process_taylor_degree=5, weights=None, rolling=None):
    # Add and process each ticker
    df, company_name = add_ticker(ticker_label, start_date, end_date)
    KNN_Weight = process_ticker(ticker_label, company_name, df, process_taylor_degree, weights, rolling)

    return np.array(KNN_Weight)

def rolling_derivatives(df, N, rolling):
    # Calculate rolling mean for the initial data
    # Calculate derivatives iteratively
    new_df = df.copy(deep=True)[-2*N:]
    # Backwards Derivative
    for i in range(N):
        new_df[f'Derivative_{i + 1}'] = new_df[f'Derivative_{i}'].diff()
    # Forwards Derivative
    for i in range(N):
        df[f'Derivative_{i + 1}'] = -df[f'Derivative_{i}'].diff(-1)
        # Drop NaN values
    df = pd.concat((df,new_df))

    if rolling is not None:
        for col in df.columns:
            df[col] = df[col].rolling(window=rolling, min_periods=1).mean() 
   
    df.dropna(inplace=True)
    return df

def create_record():
    # Create a record for all tickers
    file_path = "Stock_data/Record.csv"
    df = pd.DataFrame(columns=["Ticker", "Name", "Type", "File_path"])
    df.to_csv(file_path, index=False)

def append_to_record(Ticker, Name, File_type, File_path):
    # Append item to record
    file_path = "Stock_data/Record.csv"
    
    try:
        # Attempt to read the existing record file
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        create_record()
        df = pd.read_csv(file_path)
    
    # Ensure new record types match the existing DataFrame
    new_row = pd.DataFrame([[Ticker, Name, File_type, File_path]], 
                           columns=["Ticker", "Name", "Type", "File_path"])
    new_row = new_row.astype(df.dtypes.to_dict())

    # Check if the ticker and type already exist in the DataFrame
    mask = (df['Ticker'] == Ticker) & (df['Type'] == File_type)

    if mask.any():
        # Replace the existing row with the new record
        df.loc[mask, :] = new_row.values[0]
    else:
        # Append the new record as a new row
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to CSV
    df.to_csv(file_path, index=False)


def Get_record(Raw_data=False, Processed_data=False):
    # Get the record data
    file_path = "Stock_data/Record.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"No record found at {file_path}")
        return None
    
    if Raw_data and Processed_data:
        return df
    elif Raw_data:
        return df[df['Type'] == 'Raw']
    elif Processed_data:
        return df[df['Type'] == 'Processed']
    else:
        return df

def get_file_path(ticker):
    #retrieves file path
    file_path = "Stock_data/Record.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"No record found at {file_path}")
        return None

    file_paths = df.loc[df['Ticker'] == ticker]
    if file_paths.empty:
        raise Exception("Ticker isn't recorded")
    
    Raw_data_file_path = file_paths[file_paths['Type'] == 'Raw']['File_path'].values
    Processed_data_file_path = file_paths[file_paths['Type'] == 'Processed']['File_path'].values

    # Ensure file_paths are strings or empty
    Raw_data_file_path = Raw_data_file_path[0] if len(Raw_data_file_path) > 0 else None
    Processed_data_file_path = Processed_data_file_path[0] if len(Processed_data_file_path) > 0 else None

    return Raw_data_file_path, Processed_data_file_path

def Update_weight(ticker, weights):
    #Replaces weights
    file_path = "Stock_data/Weight_record.csv"
    

    try:
        df = pd.DataFrame({ticker: pd.Series(weights)})
        df.to_csv(file_path, index=False)
        
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame({ticker: pd.Series(weights)})
    

def get_weights(ticker):
    #retrieves weights
    file_path = "Stock_data/Weight_record.csv"    
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        return np.array(df[ticker])
    except FileNotFoundError:
        raise Exception("Ticker weights don't exist")

def plot(file_path, ax, weights = None, start=0, end=-1, raw = False, show_legend=False):
    # Create DataFrame
    if raw:
        #retrieves raw data
        df = pd.read_csv(file_path)['Open']
        color = 'b'
        label = "Open Price Raw"
    else:
        #retrieves processed data
        df = pd.read_csv(file_path)['Derivative_0']
        if weights is not None:
            df = df*weights[0]
        color = 'g'
        label = "Open Price Processed"
    
    # Extract subset of DataFrame
    if end < 1:
        X = np.arange(start,end)
        Y = df.iloc[start:end]
    else:
        X = np.arange(start,end+1)
        Y = df.iloc[start:end+1]
    
    ax.plot(X, Y, marker='o', linestyle='-', color=color, label=label, alpha=0.5)
        
    if show_legend:
        ax.legend()

