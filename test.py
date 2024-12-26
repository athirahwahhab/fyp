import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv'
df = pd.read_csv(file_path)    
# Title for the Streamlit app
st.title("Price Trend Analysis")
    
 # Show raw data if checkbox is selected
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    
item_code = 1  # AYAM STANDARD
df_item = df[df['item_code'] == item_code]  # Filter data by item code

# Group by date and calculate the average price
price_trend = df_item.groupby('date')['price'].mean()

# price trend
plt.figure(figsize=(10,6))
plt.plot(price_trend.index, price_trend.values, marker='o')
plt.title(f'Price Trend for Item Code {item_code}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
# Display the plot in Streamlit
st.pyplot(plt.gcf())
