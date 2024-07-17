import yfinance as yf
import os

# Define the tickers and their corresponding filenames
tickers = {
    "^GSPC": "sp500_data.csv",     # S&P 500
    "^DJI": "dowjones_data.csv",   # Dow Jones
    "^W5000": "wilshire5000_data.csv"  # Wilshire 5000
}
Start_date= "2020-01-01"
End_date= "2023-12-31"

# Define the relative folder path
folder_path = "Raw_data"

# Ensure the folder exists, and if not, create it
os.makedirs(folder_path, exist_ok=True)

# Download the data for each ticker and save it to a CSV file
for ticker, filename in tickers.items():
    # Define the full path for the CSV file
    file_path = os.path.join(folder_path, filename)
    
    # Remove the existing file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Download the data
    data = yf.download(ticker, start= Start_date, end= End_date)
    
    # Save the data to the specified CSV file
    data.to_csv(file_path)
    
    print(f"Data for {ticker} has been saved to {file_path}")
