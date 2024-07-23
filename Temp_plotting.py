import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "Stock_data/Raw_data/Dow Jones Industrial Average_data.csv"
df = pd.read_csv(file)

# Print columns of the DataFrame
print(df.columns)

# Selecting specific columns and manipulating dates
df = df[["Date", "Open", "Close"]]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Plotting "Open" prices over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Open"], marker='o', linestyle='-', color='b', label='Open Price', alpha= .5)
plt.plot(df.index, df["Close"], marker='o', linestyle='-', color='m', label='Close Price', alpha= .5)
plt.title('Dow Jones Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
