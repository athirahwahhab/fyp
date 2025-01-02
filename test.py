import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv'
df = pd.read_csv(file_path)    
# Title for the Streamlit app
st.title("Price Trend Analysis")

st.title("Ayam Standard")
# Filter data for item_code = 1
item_1_data = df[df['item_code'] == 1].copy()

# Convert 'date' to datetime format
item_1_data['date'] = pd.to_datetime(item_1_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_1_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_1_daily_prices = item_1_data.groupby('date')['price'].mean()

# Display the processed data
item_1_daily_prices.head(), item_1_daily_prices.info()

st.title("Daily Ayam Standard Average  ")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_1_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 1', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
# Display the plot in Streamlit
st.pyplot(plt.gcf())

st.title("Ayam Super")
item_2_data = df[df['item_code'] == 2 ].copy()

# Convert 'date' to datetime format
item_2_data['date'] = pd.to_datetime(item_2_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_2_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_2_daily_prices = item_2_data.groupby('date')['price'].mean()

# Display the processed data
item_2_daily_prices.head(), item_2_daily_prices.info()

st.title("Ayam Super Average Daily Prices ")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_2_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 2', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())

st.title("Ikan Kembung")
# Filter data for item_code = 55
item_55_data = df[df['item_code'] == 55].copy()

# Convert 'date' to datetime format
item_55_data['date'] = pd.to_datetime(item_55_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_55_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_55_daily_prices = item_55_data.groupby('date')['price'].mean()

# Display the processed data
item_55_daily_prices.head(), item_55_daily_prices.info()

st.title("Ikan Kembung Daily Average Prices")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_55_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 55', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())

st.title("Ikan Selar Kuning")
# Filter data for item_code = 69
item_69_data = df[df['item_code'] == 69].copy()

# Convert 'date' to datetime format
item_69_data['date'] = pd.to_datetime(item_69_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_69_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_69_daily_prices = item_69_data.groupby('date')['price'].mean()

# Display the processed data
item_69_daily_prices.head(), item_69_daily_prices.info()

st.title("Ikan Selar Kuning Daily Average Prices")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_69_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 69', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())

st.title("Telur Gred C")
# Filter data for item_code = 1111
item_1111_data = df[df['item_code'] == 1111].copy()

# Convert 'date' to datetime format
item_1111_data['date'] = pd.to_datetime(item_1111_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_1111_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_1111_daily_prices = item_1111_data.groupby('date')['price'].mean()

# Display the processed data
item_1111_daily_prices.head(), item_1111_daily_prices.info()

st.title("Telur Gred C Daily Average Prices")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_1111_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 1111', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())

st.title("Sayur Bayam Hijau")
# Filter data for item_code = 1556
item_1556_data = df[df['item_code'] == 1556].copy()

# Convert 'date' to datetime format
item_1556_data['date'] = pd.to_datetime(item_1556_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_1556_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_1556_daily_prices = item_1556_data.groupby('date')['price'].mean()

# Display the processed data
item_1556_daily_prices.head(), item_1556_daily_prices.info()

st.title("Sayur Bayam Hijau Daily Average Prices")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_1556_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 1', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())

st.title("Sayur Sawi Hijau")
# Filter data for item_code = 1558
item_1558_data = df[df['item_code'] == 1558].copy()

# Convert 'date' to datetime format
item_1558_data['date'] = pd.to_datetime(item_1558_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_1558_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_1558_daily_prices = item_1558_data.groupby('date')['price'].mean()

# Display the processed data
item_1558_daily_prices.head(), item_1558_daily_prices.info()

st.title("Sayur Sawi Hijau Daily Average Prices ")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_1558_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 1558', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())
