
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = '/content/combined_output_latest.csv'
df = pd.read_csv(file_path)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title for the Streamlit app
st.title("Price Trend Analysis")
item_code = 1  # AYAM STANDARD

# Filter data by item code
df_item = df[df['item_code'] == item_code]

# Group by date and calculate the average price
price_trend = df_item.groupby('date')['price'].mean()

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(price_trend.index, price_trend.values, marker='o')
ax.set_title(f'Price Trend for Item Code {item_code}')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)

