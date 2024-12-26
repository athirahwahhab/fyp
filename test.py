
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data(nrows):
    data = pd.read_csv('combined_filtered_allyears', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

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

