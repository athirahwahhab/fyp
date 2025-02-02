import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

st.title("Telur Gred C")
multi = '''Egg grade C is a grade of chicken egg that weighs between 55.00 to 59.9 grams per egg. 
This type of egg are more affordable and easier to find than eggs grade A and B(Hazim, 2022).
'''
st.markdown(multi)
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
multi = '''From July 2023 to October 2023, the average daily price stayed high, around RM12.20. 
During this time, there were very small changes. In November 2023, prices went down a little, falling just below RM12.20.
This decrease happened slowly and was probably a normal market adjustment. 
In April 2024, prices dropped going from about RM12.10 to around RM11.50 in a short time. 
This shows that something big happened in the market such as much supply, less demand, or economic issues.
In May 2024, the price dropped to the lowest price of eggs which is RM11.40, showing the biggest fall during the time we looked at. 
This decrease might be due to important changes in the market or outside factors. 
By the end of May 2024, prices started to slowly up again, reaching around RM 11.80 by the end of June 2024.
This shows that things were stabilizing after the big drop. 
Even though prices were recovering, the price still go up and down a little in June 2024, staying between RM 11.75 and RM 11.85. 
Overall, the graph shows that the market was still unstable.
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
multi = '''From July 2023 to February 2024, the price stayed mostly the same about RM12.30. 
This long period of little change shows that the supply and demand for eggs were balanced during this time. 
In March 2024, the price began to go down a little, falling to around RM12.25.
This marks the start of a downward trend after many months of stable prices. 
In April 2024, there was a huge fall in prices, dropping to around RM12.00. 
This shows maybe a big market problem likely caused by economic shifts, less demand, or too much supply.
By May 2024, the price dropped even more , hitting the lowest point at about RM11.95. 
It stayed at this time in June 2024, making it the lowest average price for any month during the time we looked at.
'''
st.markdown(multi)
# Show the plot
plt.show()
st.pyplot(plt.gcf())

st.title("Diagnostic Analysis")

multi = ''' Supply Chain is one of the factors because the price of remained relatively stable because the Malaysian government decided to continue subsidies and price control for Grade A, B, and C eggs(KPDNK, 2023). 
This subsidy extension helped stabilize egg prices nationwide, ensuring affordability. 
Addtionally, the Agro Madani Sales intiative in Kelantan provided temporary relief by offering affordable goods, which helped maintain price stability even during the flood season(Awani, 2023). 
Between May and June 2024, the government reduced the retail price of Grade A, B, and C eggs by 3 cent per egg. 
This decision was made due to lower input costs for egg production, particularly animal feed, and aimed to pass targeted subsidy savings back to the public. 
The subsidy initiative involved RM100 million for eggs, which eased production costs and stabilized food prices across the nation(Muhammad Yusri Muzamir, 2024).
'''
st.markdown(multi)
multi = ''' Climate change also had a impact in the price trend particularly through the monsoon season. 
The end-of-year flooding in Kelantan disrupted the chicken supply chain, affecting market operations and make prices increase. 
The seasonal flooding typically hinders production and distribution, further complicating the supply of chicken(NADMA, 2023).
'''
st.markdown(multi)

multi = '''Market Fundamentals and Demand also one of the factors increasing the egg prices. 
In March 2024, the price of grade C eggs was remained stable at RM12.30, because it starts of Ramadhan.
During this period, the demand for eggs was high as the Muslim population, which forms the majority in Malaysia, prepared meals for sahur and iftar. 
This high demand likely maintained price stability despite other market factor. 
Between April and June 2024, the price of eggs dropped sharply from RM12.30 in March to RM12.00 in April and continued decreasing to RM11.95 
by June 2024, the price decrease was likely due to reduced demand following the Raya celebrations, while the gradual decline between May and June was influenced by the government subsidy adjustment(Patah, 2023).
'''
st.markdown(multi)
