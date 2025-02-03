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
                                                                                                
st.title("Sayur Sawi Hijau")
multi = '''Sawi hijau is  mustard green that have a peppery-tasting green that come from a mustard 
plant(Frey, 2020). This can be cooked in boiled, steamed and stir fried and mustard green is the 
70 
nutritious food and have low calories in the vegetable(Frey, 2020). Mustard greens are low in 
calories yet high in fibre and many essential vitamins and minerals. In particular, they’re an 
excellent source of vitamins C and K(Frey, 2020). This vegetable also is rich in important plant 
compounds and micronutrients, specifically vitamins A, C, and K. As a result, eating them may 
have benefits for eye and heart health, as well as anticancer and immune-boosting 
properties(Frey, 2020).  
'''
st.markdown(multi)
# Filter data for item_code = 1558
item_1558_data = df[df['item_code'] == 1558].copy()

# Convert 'date' to datetime format
item_1558_data['date'] = pd.to_datetime(item_1558_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_1558_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
item_1558_daily_prices = item_1558_data.groupby('date')['price'].mean()

# Display the processed data
item_1558_daily_prices.head(), item_1558_daily_prices.info()

st.title("Sayur Sawi Hijau Daily Average Prices ")
# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(item_1558_daily_prices, label='Daily Average Price', color='blue')
plt.title('Daily Average Prices of Item 1558', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt.gcf())
multi = '''In July 2023, the price began at a high point, going above RM8.00 per kilogram. This 
shows that there was either a lot of demand or not enough supply at the start. However, the 
price then dropped quickly, falling to around RM4.50 by August 2023. From September to 
December 2023, the price went up and down slightly but stayed mostly maintained between 
RM4.50 and RM5.50. This suggests the market was balancing out after the big drop earlier. In 
January 2024, the prices started to fall, reaching their lowest point of about RM4.00 by March 
2024. This decrease shows that demand was getting weak or there was too much supply during 
this time. In May 2024, there was a big jump in prices, going over RM8.00. this was the highest 71 
price seen during the time we looked at. The quick rise might have been because of the time of 
year, less stuff being available, or more people wanting it. After hitting the highest point in 
May, prices fell quickly and went back to about RM5.00 by July 2024. This shows that the 
market was adjusting or going back to normal after the big increase. 
'''
st.markdown(multi)
import matplotlib.pyplot as plt

# Filter the data for a specific item, e.g., item_code 1558
item_code_to_check = 1558
filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

st.title("Sayur Sawi Average Monthly Prices")

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

multi = '''In July 2023, the price was quite high, going above RM5.00, but it dropped sharply by 
August 2023, falling to around RM4.50. This shows that there was some instability at first, 
maybe because of some changes in the market. From September to December 2023, the prices 
stayed mostly steady, moving between RM4.50 and RM5.50. This means there was a time when 
supply and demand were balanced. In January and February 2024, the prices kept going down, 
hitting the lowest point of RM4.00 in March 2024, this drop might be because there was too 
much supply or less demand during those months. In April 2024 there was a big jump in prices, 
reaching their highest point at about RM7.50 in May 2024. This sudden rise probably happened 
because of major market problems, like high demand during certain time. After that, prices are 
feeling to RM 6.00 in June 2024.  
'''
st.markdown(multi)

st.title("Diagnostic Analysis")

multi = ''' Supply chain is one the factors because from January to March 2024, the Malaysian 
government launched the Program Jualan Rahmah Madani on January 11 2023. This year 
around initiative by the Ministry of Domestic Trade and Cost of Living (KPDN) provided 
essential goods, including vegetables  at 10% to 30% lower prices, especially in low-income 
areas(Muhafandi Muhammad, 2023). The program helped reduce market dependency by 
encouraging affordability and ensuring a stable supply during many period such as preparation 
before  Ramadhan (Bernama, 2024) and preparation for monsoon season(Bernama, 2024). 
'''
st.markdown(multi)
multi = ''' September 2023, the monsoon season brought heavy rains that disrupted 
vegetable farming and reduced crop yields contributing to a gradual increase in the price of 
vegetable(Mukhriz Hazim & Yap Si Err, 2024).In December 2023, prolonged rains caused 
signicant crop damage, as reported in Kelantan markets such as RTC Tunjong.(Hidayatidayu 
Razali, 2023c). In May 2024. Dry weather condition affected agricultural yields leading to 
higher prices. Report highlighted that prolonged drought caused wells to dry up and crops to 
fail, making agricultural challenges(Hazelen Liana Kamarudin, 2024). 
'''
st.markdown(multi)

multi = '''  In April 2024,  during the festive season of Eid al-Fitri, demand for vegetables, 
including green mustard, rise significantly for celebration preparations. Although the 
government implemented(KPDNK, 2024b) a festive price control scheme for ensure 
affordability, overall price still increased due to high demand for festive cooking. In May 2024, 
the high prices due to the spill over effect from the festive season, coupled with lower yields 
caused by dry weather(Hazelen Liana Kamarudin, 2024). By June 2024, after the post-festive 
time, consumer demand reduced leading to a drop in prices(Ercy Gracella Ajos, 2023). 
'''
st.markdown(multi)
