import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

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

st.title("Ayam Standard Average Daily Prices ")
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
# Convert the 'date' column to datetime, trying 'dayfirst=True' in case of inconsistent date formats
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Create a new column 'year_month' for grouping by month
df['year_month'] = df['date'].dt.to_period('M')

# Group by item_code and year_month, and calculate the average price
average_price_per_item_monthly = df.groupby(['item_code', 'year_month'])['price'].mean().reset_index()

# Display the first few rows to check the result
average_price_per_item_monthly.head()
import matplotlib.pyplot as plt

# Filter the data for a specific item, e.g., item_code 1
item_code_to_check = 1
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

st.title("Ayam Standard Average Monthly Prices")

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
# Display the plot in Streamlit
st.pyplot(plt.gcf())

multi = '''The price of “Ayam standard” in Kelantan went up and down a lot from June 2023 to June 2024. The highest price was around RM9.30 per 
kilogram(kg) in June 2023, but by August 2023 it dropped and it became the lowest price that it was about RM 8.10 in August 2023. 
After that, the price started to slowly go up again. From September to December 2023,
the price stayed mostly steady, moving between RM8.20 and RM 8.60. As it entered 2024, the price changes a little.
In January it went up a bit but then dropped a bit in February. By March 2024 the price increased again, showing a steady improvement. 
Starting in April 2024, the price has been rising again, reaching RM 9.30 in June 2024. 
This price matched the highest price at the beginning of the recorded price in June 2023.
'''
st.markdown(multi)
