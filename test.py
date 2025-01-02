import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv'
df = pd.read_csv(file_path)    
# Title for the Streamlit app
st.title("Price Trend Analysis")
st.title("Ayam Standard ")
item_code = 1  # AYAM STANDARD
df_item = df[df['item_code'] == item_code]  # Filter data by item code

# Group by date and calculate the average price
price_trend = df_item.groupby('date')['price'].mean()

# price trend
plt.figure(figsize=(10,6))
plt.plot(price_trend.index, price_trend.values, marker='o')
plt.title(f'Price Trend for Item Code {item_code}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
# Display the plot in Streamlit
st.pyplot(plt.gcf())

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

