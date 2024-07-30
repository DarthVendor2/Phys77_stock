import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class Data_handler:
    def __init__(self):
        # Sets folder paths
        self.stock_folder= "Stock_data"
        self.raw_data_folder_path = "Stock_data/Raw_data"
        self.processed_data_folder_path = "Stock_data/Processed_data"
        os.makedirs(self.raw_data_folder_path, exist_ok=True)
        os.makedirs(self.processed_data_folder_path, exist_ok=True)

    def Get_record(Raw_data= False, Processed_data= False):
        pass
        

    def add_ticker(self, ticker_label, start_date="2000-01-01", end_date="2023-12-31"):
        try:
            ticker = yf.Ticker(ticker_label)
            company_name = ticker.info.get('longName', ticker_label)
            data = yf.download(ticker_label, start=start_date, end=end_date)
            
            file_path = os.path.join(self.raw_data_folder_path, f'{company_name}_data.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
            
            data.to_csv(file_path)

        except Exception as e:
            raise f"Error adding ticker {ticker_label}: {e}"

    def process_tickers(self, Degree_of_taylor, rolling=30):

        Degree_of_taylor += 1
        files = os.listdir(self.raw_data_folder_path)
        for file in files:
            file_path = os.path.join(self.raw_data_folder_path, file)
            processed_file_path = os.path.join(self.processed_data_folder_path, file)
            
            if os.path.exists(processed_file_path):
                os.remove(processed_file_path)
            
            try:
                df = pd.read_csv(file_path)[['Open']]
                df.rename(columns={'Open': 'Derivative_0'}, inplace=True)
                df['Derivative_0'] = df['Derivative_0'].rolling(rolling).mean()

                for i in range(Degree_of_taylor):
                    df[f'Derivative_{i + 1}'] = -df[f'Derivative_{i}'].diff(-1)

                for col in df.columns[1:]:
                    df[col] = df[col].rolling(rolling).mean()

                df.dropna(inplace=True)
                df.to_csv(processed_file_path, index=False)

            except Exception as e:
                raise f"Error processing file {file}: {e}"

class Plotter:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)[["Open"]]

    def plot(self, ax, start=0, end=-1, show_legend=False):
        """
        Plot a subset of the 'Open' data from the DataFrame.

        :param ax: Matplotlib axes to plot on.
        :param start: Starting index for plotting (inclusive).
        :param end: Ending index for plotting (exclusive). If -1, plots until the end.
        :param show_legend: Whether to show the legend in the plot.
        """
        # Ensure start and end are scalar values
        if isinstance(start, (np.ndarray, list)):
            start = start[0]  # or some other logic to handle arrays/lists
        if isinstance(end, (np.ndarray, list)):
            end = end[0]  # or some other logic to handle arrays/lists

        # Adjust end index if it's -1 to include all rows from start to the end of the DataFrame
        if end == -1 or end > len(self.df):
            end = len(self.df)

        # Ensure start and end are within valid range
        start = max(int(start), 0)
        end = min(int(end), len(self.df))

        # Extract subset of DataFrame
        df = self.df.iloc[start:end]

        print(f"Plotting data from index {start} to {end}")

        N = len(df)
        X = np.linspace(0, N, N)
        ax.plot(X, df, marker='o', linestyle='-', color='b', label='Open Price', alpha=0.5)
        
        if show_legend:
            ax.legend()


