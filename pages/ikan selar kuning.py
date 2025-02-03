import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path) 

# Convert the 'date' column to datetime, trying 'dayfirst=True' in case of inconsistent date formats
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Create a new column 'year_month' for grouping by month
df['year_month'] = df['date'].dt.to_period('M')

# Group by item_code and year_month, and calculate the average price
average_price_per_item_monthly = df.groupby(['item_code', 'year_month'])['price'].mean().reset_index()

# Display the first few rows to check the result
average_price_per_item_monthly.head()
import matplotlib.pyplot as plt

st.title("Ikan Selar Kuning")
multi = '''“Ikan selar kuning” is a yellowstripe scad that is a small and common fish in Malaysia. 
It is known for its mild flavour and tender texture, making it a way to cook such as frying and 
grilling. The fish has a streamlined body with a distinct yellow stripe along its sides and 
typically measures between 10 to 20 cm in length. Rich in nutrients, it provides a high protein 
content essential for muscle repair and overall health, along with omega-3 fatty acids that 
improving heart health(Juhari, 2024).
'''
st.markdown(multi)
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

multi = '''In July 2023, prices started at a high level, averaging around RM13.50 to RM14.00. 
These high prices stayed mostly steady, with small changes until September 2023, showing a 
period of stable but high costs. In October 2023, there was a sudden and large drop in prices, 
with the average fall to about RM10.00, the lowest point of the year 2023. This hard decrease 
might be due to big changes in the market, like more supply, less demand, or external issues 
such as new policies. In January 2024, the price increased  to about RM14.00. After hitting a 
high value in January, prices started to drop slowly, reaching an average of around RM11.00 
by April 2024. This decline shows that either demand went down or there was too much supply. 
April 2024 is the lowest price with prices close to RM11.00. From May 2024, prices began to 
rise a little, between RM11.50 and RM12.00 by June 2024. While this shows some 
improvement, prices were still much lower than the last year mid of 2023. 
'''
st.markdown(multi)
st.title("Ikan Selar Kuning Monthly Average Prices")
# Convert the 'date' column to datetime, trying 'dayfirst=True' in case of inconsistent date formats
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Create a new column 'year_month' for grouping by month
df['year_month'] = df['date'].dt.to_period('M')

# Group by item_code and year_month, and calculate the average price
average_price_per_item_monthly = df.groupby(['item_code', 'year_month'])['price'].mean().reset_index()

# Display the first few rows to check the result
average_price_per_item_monthly.head()
import matplotlib.pyplot as plt

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

multi = '''The price started high in June 2023, averaging RM12.50, which shows that there was 
either a lot of demand or not enough supply. The price stayed high in July 2023, with only a 
small drop. From August 2023, prices started to go down slowly reaching around RM11.50 by 
October 2023. This steady drop likely happened because of changes in the market, like more 
supply or less demand. In November 2023, prices fell sharply to an average of about RM10.50, 
the lowest point during this period. This sudden drop could be because of market issues such 
as too much supply or less demand. After a big drop in November, prices returned in January 
2024, reaching an average of RM12.50. After hitting the highest point in January, prices started 
to drop slowly again, going down from RM12.00 to about RM10.80 in April 2024. This shows 
a steady decrease in demand or too much supply in the market. In May 2024, prices began to 
rise a little, reaching RM11.00, and this small increase continued into June 2024, with prices 
staying around RM11.20. However, the recovery wasn’t strong enough to get the earlier high 
levels. 
'''
st.markdown(multi)

st.title("Diagnostic Analysis")

multi = ''' The supply Chain is one of the factors because there is a phenomenon that called “bulan 
cerah” in June 2023 that reduced fish catches due to jellyfish infestations and unfavourable sea 
conditions. This phenomenon causes a limited in supply of fresh fish and it makes the prices 
higher as shown in graph the start of graph it starting higher prices (Hidayatidayu Razali, 2023). 
Next, the monsoon season at December 2023 caused a significant reduction in fishing activities, 
further disrupting the supply chain. Traders reported higher wholesale costs, leading to retail 
price increases for fish. By March 2024, frozen imported fish entered the market helping to 
stabilize prices which is shown in the graph the price started to decrease.
'''
st.markdown(multi)
multi = '''Economic Factors also played a crucial role. The Program Jualan Rahmah started in 
Kelantan aimed to provide essential goods at 10% to 30% below also helping to reduce price 
spikes as shown in the graph when the price started to decrease(KPDNK, 2024). Additionally, 
higher whole casts costs during monsoon season and adjustments to supply and demand 
influenced fish prices. For instance, in November 2023, abundant fish supply reduced prices, 
as shown in the graph that the price is the lowest at that time(Rosliza Mohamed, 2023). 
However, in December 2023, limited supplies caused prices to increase again(Zuliaty Zulkiffli, 
2024). 
'''
st.markdown(multi)

multi = '''Climate changes had an impact on fish prices. Reports show that climate-related 
changes have led to a 60% decrease in caught fish, resulting in a significant reduction in supply. 
This was shown in late 2023 limited fishing activities due to rough seas contributed to supply 
shortages and rising prices(Zuliaty Zulkiffli, 2024). Also, the “bulan cerah’ phenomenon in 
80 
mid-2023 marked by hot weather and jellyfish infestation disrupted fishing activities 
(Hidayatidayu Razali, 2023). 
'''
st.markdown(multi)
