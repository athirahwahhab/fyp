import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

st.title("Sayur Bayam Hijau")
multi = '''Bayam hijau is a green spinach that is a leafy vegetable widely consumed in Malaysia and 
Indonesia. This spinach is versatile with varieties in green and red colour and it can be eaten raw, 
steamed or stir fried as daily food(GardeningSG, 2023). 
'''
st.markdown(multi)
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

multi = '''In July 2023, prices were very unstable, moving between RM6.00 and RM7.00. Then, 
there was a sudden jump above RM8.00, but this didn’t last long. By August 2023, prices fell 
below RM6.00. From September to November 2023, prices slowly went up and settled around 
RM7.00 by November. This suggests the market was balancing itself or that more people were 
buying.  In January 2024, prices high up to their highest point, reaching around RM8.50. This 
big increase might have been caused by seasonal reasons, less supply, or more demand. After 
hitting the peak in January, prices started to drop to around RM6.50 by April 2024.  From May 
to June 2024, prices changed a little, staying between RM6.00 and RM6.50. These small 
changes show that the market is still adjusting, but overall, prices stayed the same compared to 
before.  
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

# Filter the data for a specific item, e.g., item_code 1556
item_code_to_check = 1556
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]
st.title("Sayur Bayam Hijau Average Monthly Prices")
# Plotting the price trend for the specific item
plt.figure(figsize=(10, 6))
plt.plot(filtered_item_data['year_month'].astype(str), filtered_item_data['price'], marker='o', linestyle='-', color='g')
plt.title(f'Price Trend for Item {item_code_to_check}')
plt.xlabel('Month')
plt.ylabel('Average Price (RM)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
st.pyplot(plt.gcf())
multi = '''From July to September 2023, prices stayed mostly steady, moving between RM5.50 
and RM6.00. This shows the market was balanced, with little change. From October to 
November 2023, prices slowly increased, from RM 6.00 to around RM6.50. This gradual 
increase might be because of higher demand or lower supply. In December 2023, prices jumped 
sharply to RM8.00 which is the highest point. After reaching the highest point in December, 
prices started to drop and went down to RM6.00 by March 2024. Between April and June 2024, 
prices went up and down slightly, staying around RM6.00 to RM 6.50. This shows a time of 
more stable prices after the big ups and downs in late 2023 and early 2024. 
'''
st.markdown(multi)

st.title("Diagnostic Analysis")

multi = ''' In December 2023, the price increase for increase for green spinach was largely due to 
a reduced supply caused by prolonged rainfall, which impacted local vegetable. Wholesalers 
had to rely on imports from Thailand and Vietnam to meet demand, increasing costs and leading 
farmers and traders to raise prices. This shortage and reliance on imported spinach were key 
contributors to the upward price trend during this period (Hidayatidayu Razali, 2023c). This 
69 
show that supply chain is one of the factors that contributing the price trend. Also, between 
January and March 2024, the price of green spinach in Kelantan dropped because there was too 
much of it available. This happened because more people starting growing their vegetables at 
home, the encouragement from the Consumer’s Association of Penang (CAP). As a result, 
fewer people bought spinach from the market , and the extra supply , along with less demand, 
caused the prices to go down during the months(Suraya Ali, 2023). 
'''
st.markdown(multi)
multi = ''' The monsoon season in November 2023 cause of climate change brought continuous 
heavy rains, which damaged over one million kilograms of vegetables in southern Peninsular 
Malaysia. This created a shortage of vegetable led to a significant price increase(Hidayatidayu 
Razali, 2023). 
'''
st.markdown(multi)

multi = ''' For market fundamentals and demand in October 2023, the price of green spinach 
decreases much of it available. This happened due to good weather, which helped farmers grow 
more vegetable than people needed. As a result, farmers had to sell spinach for less than it cost 
to grow, just to avoid having too much left over. Even though they lost money farmers kept 
selling to make some income and some got help from the Federal Agricultural Authority 
(FAMA)(Ercy Gracella Ajos, 2023). 
'''
st.markdown(multi)
