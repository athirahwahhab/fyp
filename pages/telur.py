import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv'
df = pd.read_csv(file_path)   

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
