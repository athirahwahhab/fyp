import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

st.title("Ayam Super")
st.markdown(multi)
multi = ''' Ayam Super is a whole chicken sold without the head, feet, liver, and gizzard. This makes it more processed and convenient for cooking, catering to consumers who prefer less preparation work. 
As of June 2022, the retail price for Ayam Super was capped at RM9.90 per kilogram in Peninsular Malaysia under the same government price control initiative. 
Ayam Super is slightly more expensive than Ayam Standard due to the additional cleaning and processing involved, making it a preferred choice for those seeking convenience.
'''
st.markdown(multi)
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

st.markdown(multi)
multi = '''At the beginning of July 2023, the price start was quite high, around RM10.00 per kilogram. 
Throughout the month, the price stayed stable at around RM 10.00 before it went down. 
In August 2023, there was a sudden drop in price falling about RM 8.30 which was the lowest point during this time.
From September to November 2023, prices slowly began to rise again, reaching about RM 9.50 by November. 
Between January and March 2024, prices went up and down a lot, moving between RM8.00 and RM 9.50 without a clear direction. 
This time was full of sudden and frequent price changes, possibly because of issues with supply and demand or other factors affecting the market. 
The most unstable time was between March and May 2024. Prices kept going up above RM 9.50 and then dropping below RM8.50. 
From the end of May 2024, prices started going up again and reached their highest point at around RM 10.00 in June 2024. 
This was the highest price since July 2023. In July 2024, prices kept up and down between RM9.00 and RM9.50. 
This shows things were a bit more stable but still changing a lot.
'''
st.markdown(multi)
# Convert the 'date' column to datetime, trying 'dayfirst=True' in case of inconsistent date formats
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Create a new column 'year_month' for grouping by month
df['year_month'] = df['date'].dt.to_period('M')

# Group by item_code and year_month, and calculate the average price
average_price_per_item_monthly = df.groupby(['item_code', 'year_month'])['price'].mean().reset_index()

# Display the first few rows to check the result
average_price_per_item_monthly.head()
import matplotlib.pyplot as plt

st.title("Ayam Super Average Monthly Prices")
# Filter the data for a specific item, e.g., item_code 2
item_code_to_check = 2
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

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
st.markdown(multi)
multi = '''In July 2023, the price was approximately RM 9.75 per kilogram, which is one of the highest monthly averages during this time.
This shows a strong market value at the beginning of the analysis. 
Next, in August 2023 the price dropped sharply to the lowest average of about RM 8.25 per kilogram. 
This was the biggest drop during the entire period. After the big drop in August, the price rising back in September 2023 to around RM9.25.
From October to December 2023, prices stayed steady at around RM9.00, with very little change. 
This shows the market was balanced in the last few months of the year. 
From January to April 2024, the price stayed mostly stable at around RM9.00. 
However, there were small drops in February and April bringing the average price down to RM8.75 during those months. 
This time showed some ups and downs but no big problems. In May 2024, prices began to go up again, reaching about RM9.25 by June 2024. 
Overall the prices went through quick drops and slow changes, showing shifts in the market.
'''
st.markdown(multi)
