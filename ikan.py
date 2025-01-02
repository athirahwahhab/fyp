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
