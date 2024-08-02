import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def add_ticker(ticker_label, start_date, end_date):
    """
    Adds a new ticker to the record by downloading historical data and saving it to a CSV file.

    :param ticker_label: The ticker symbol of the stock.
    :param start_date: The start date for downloading historical data.
    :param end_date: The end date for downloading historical data.
    :return: DataFrame containing the downloaded data and the company name.
    """
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

def process_ticker(ticker, company_name, df, Full_taylor_degree, rolling):
    """
    Processes a single ticker's data by calculating rolling derivatives and saving it to a CSV file.

    :param ticker: The ticker symbol of the stock.
    :param company_name: The name of the company.
    :param df: DataFrame containing the raw data.
    :param Full_taylor_degree: The degree for calculating derivatives.
    :param rolling: The window size for rolling calculations.
    :return: KNN Weights calculated from the processed data.
    """
    Processed_data_folder_path = "Stock_data/Processed_data"
    file_path = os.path.join(Processed_data_folder_path, f'Processed_{company_name}_data.csv')

    try:
        # Extract 'Open' prices and rename column
        df = df["Open"].to_frame()
        df.rename(columns={'Open': 'Derivative_0'}, inplace=True)

        # Calculate rolling derivatives
        df, KNN_Weights = rolling_derivatives(df, Full_taylor_degree, rolling)

        # Save processed data
        df.to_csv(file_path, index=False)

        # Append to record
        append_to_record(ticker, company_name, "Processed", file_path)

        return KNN_Weights

    except Exception as e:
        raise Exception(f"Error processing ticker {ticker}: {e}")

def add_and_process_tickers(ticker_labels, start_date="2000-01-01", end_date="2023-12-31", Full_taylor_degree=5, rolling=30):
    """
    Adds and processes multiple tickers, storing their KNN Weights.

    :param ticker_labels: List of ticker symbols of the stocks.
    :param start_date: The start date for downloading historical data.
    :param end_date: The end date for downloading historical data.
    :param Full_taylor_degree: The degree for calculating derivatives.
    :param rolling: The window size for rolling calculations.
    :return: Numpy array of KNN Weights for each ticker.
    """
    Weights = []
    for ticker_label in ticker_labels:
        # Add and process each ticker
        df, company_name = add_ticker(ticker_label, start_date, end_date)
        KNN_Weight = process_ticker(ticker_label, company_name, df, Full_taylor_degree, rolling)
        Weights.append(KNN_Weight)
    return np.array(Weights)

def rolling_derivatives(df, N, rolling=30):
    """
    Calculates rolling derivatives of the DataFrame.

    :param df: DataFrame containing the data.
    :param N: The degree for calculating derivatives.
    :param rolling: The window size for rolling calculations.
    :return: DataFrame with rolling derivatives and KNN Weights.
    """
    # Calculate rolling mean for the initial data
    df['Derivative_0'] = df['Derivative_0'].rolling(rolling).mean()

    # Calculate derivatives iteratively
    for i in range(N):
        df[f'Derivative_{i + 1}'] = -df[f'Derivative_{i}'].diff(-1)

    # Apply rolling mean to derivatives
    for col in df.columns[1:]:
        df[col] = df[col].rolling(rolling).mean()

    # Drop NaN values
    df.dropna(inplace=True)

    # Calculate KNN Weights
    KNN_Weight = df.iloc[0]
    df = df / df.iloc[0]

    return df, KNN_Weight

def create_record():
    """
    Creates a new CSV file to record ticker information.
    """
    file_path = "Stock_data/Record.csv"
    df = pd.DataFrame(columns=["Ticker", "Name", "Type", "File_path"])
    df.to_csv(file_path, index=False)

def append_to_record(Ticker, Name, File_type, File_path):
    """
    Appends a new record to the existing record CSV file.

    :param Ticker: The ticker symbol of the stock.
    :param Name: The name of the company.
    :param Type: The type of data (Raw or Processed).
    :param File_path: The file path where the data is saved.
    """
    file_path = "Stock_data/Record.csv"
    # Read the existing record
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        create_record()
        df = pd.read_csv(file_path)
    
    
    # Append the new record
    new_row= [Ticker, Name, File_type, File_path]
    df.loc[len(df.index)] = new_row
    df.to_csv(file_path, index=False)



def Get_record(Raw_data=False, Processed_data=False):
    """
    Retrieves records from the CSV file based on the specified data type.

    :param Raw_data: Whether to retrieve raw data records.
    :param Processed_data: Whether to retrieve processed data records.
    :return: DataFrame containing the requested records.
    """
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

def append_weight(ticker, weights):
    """
    Appends KNN Weights to a CSV file.

    :param ticker: The ticker symbol of the stock.
    :param weights: The KNN Weights to be appended.
    """
    file_path = "Stock_data/Weight_record.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Append weights to the DataFrame
    df[ticker] = pd.Series(weights)
    df.to_csv(file_path, index=False)

def plot(file_path, ax, start=0, end=-1, show_legend=False):
    """
    Plot a subset of the 'Open' data from the DataFrame.

    :param file_path: Path to the CSV file.
    :param ax: Matplotlib axes to plot on.
    :param start: Starting index for plotting (inclusive).
    :param end: Ending index for plotting (exclusive). If -1, plots until the end.
    :param show_legend: Whether to show the legend in the plot.
    """
    df = pd.read_csv(file_path)['Open']

    # Ensure start and end are scalar values
    if isinstance(start, (np.ndarray, list)):
        start = start[0]
    if isinstance(end, (np.ndarray, list)):
        end = end[0]

    if end == -1 or end > len(df):
        end = len(df)

    start = max(int(start), 0)
    end = min(int(end), len(df))

    df = df.iloc[start:end]

    print(f"Plotting data from index {start} to {end}")

    N = len(df)
    X = np.linspace(0, N-1, N)
    ax.plot(X, df, marker='o', linestyle='-', color='b', label='Open Price', alpha=0.5)
        
    if show_legend:
        ax.legend()
