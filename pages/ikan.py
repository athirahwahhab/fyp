import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv'
df = pd.read_csv(file_path) 
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

# Filter the data for a specific item, e.g., item_code 1
item_code_to_check = 55.0
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

st.title("Ikan Kembung Average Monthly Prices")

# Plotting the price trend for the specific item
plt.figure(figsize=(10, 6))
plt.plot(filtered_item_data['year_month'].astype(str), filtered_item_data['price'], marker='o', linestyle='-', color='g')
plt.title(f'Price Trend for Item {item_code_to_check} Monthly Average')
plt.xlabel('Month')
plt.ylabel('Average Price (RM)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
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

st.title("Ikan Selar Kuning Monthly Average Prices")
# Filter the data for a specific item, e.g., item_code 69
item_code_to_check = 69
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

# Plotting the price trend for the specific item
plt.figure(figsize=(10, 6))
plt.plot(filtered_item_data['year_month'].astype(str), filtered_item_data['price'], marker='o', linestyle='-', color='g')
plt.title(f'Price Trend for Item {item_code_to_check} Monthly Prices')
plt.xlabel('Month')
plt.ylabel('Average Price (RM)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
st.pyplot(plt.gcf())
