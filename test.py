import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data(nrows):
    data = pd.read_csv('fyp/data/combined_filtered_allyears .csv', nrows=nrows)
    return data

try:
    # You can adjust the number of rows as needed
    data = load_data(nrows=None)  # None to load all rows
    
    # Title for the Streamlit app
    st.title("Price Trend Analysis")
    
    # Show raw data if checkbox is selected
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    # Add item selection dropdown
    available_items = data['item_code'].unique()
    item_code = st.selectbox('Select Item Code', available_items, index=0)
    
    # Filter data by item code
    df_item = data[data['item_code'] == item_code]
    
    # Group by date and calculate the average price
    price_trend = df_item.groupby('date')['price'].mean()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(price_trend.index, price_trend.values, marker='o')
    ax.set_title(f'Price Trend for Item Code {item_code}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    plt.xticks(rotation=45)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
