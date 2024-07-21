import yfinance as yf
import pandas as pd
import os


class data_handler:
    def __init__(self):
        # Sets folder paths
        self.raw_data_folder_path = "Stock_data/Raw_data"
        self.processed_data_folder_path = "Stock_data/Processed_data"
        os.makedirs(self.raw_data_folder_path, exist_ok=True)
        os.makedirs(self.processed_data_folder_path, exist_ok=True)

    def add_ticker(self, ticker_label, start_date="2000-01-01", end_date="2023-12-31"):
        # Get company name from ticker
        ticker = yf.Ticker(ticker_label)
        company_name = ticker.info['longName']

        # Download data
        data = yf.download(ticker_label, start=start_date, end=end_date)

        # Define file path for saving
        file_path = os.path.join(self.raw_data_folder_path, f'{company_name}_data.csv')

        # Remove the existing file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Save the data to CSV
        data.to_csv(file_path)
        
        print(f"Data for {ticker_label} has been saved to {file_path}")

    def process_tickers(self, Degree_of_taylor, rolling=30):
        Degree_of_taylor+= 1
        files = os.listdir(self.raw_data_folder_path)
        for file in files:
            file_path = os.path.join(self.raw_data_folder_path, file)
            processed_file_path = os.path.join(self.processed_data_folder_path, file)
            
            # Remove the original file if needed
            if os.path.exists(processed_file_path):
                print(f'Removing {processed_file_path}')
                os.remove(processed_file_path)
            
            # Only take the opening value for each day
            df = pd.read_csv(file_path)[['Open']]
            df.rename(columns={'Open': 'Derivative_0'}, inplace=True)

            # Calculate N number of derivatives
            for i in range(Degree_of_taylor):
                df['Derivative_' + str(i + 1)] = -df[f'Derivative_{i}'].diff(-1).rolling(rolling).mean()
            
            # Drop rows with NaN values (incomplete rows due to differencing)
            df.dropna(inplace=True)

            # Save the data to the specified CSV file in the processed folder
            df.to_csv(processed_file_path, index=False)
    
            print(f"Data has been processed and saved for {file}")

