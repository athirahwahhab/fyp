import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
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

# Convert the 'date' column to datetime, trying 'dayfirst=True' in case of inconsistent date formats
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Create a new column 'year_month' for grouping by month
df['year_month'] = df['date'].dt.to_period('M')

# Group by item_code and year_month, and calculate the average price
average_price_per_item_monthly = df.groupby(['item_code', 'year_month'])['price'].mean().reset_index()

# Display the first few rows to check the result
average_price_per_item_monthly.head()

item_code_to_check = 1111.0

# Filter the data for the specific item
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

st.title("Telur Gred C Average Monthly Prices")

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
