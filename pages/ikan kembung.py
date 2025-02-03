import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path) 

st.title("Ikan Kembung")
multi = '''Ikan kembung is a Indian mackerel that is a nutritious and affordable local fish in 
Malaysia that rich in omega-3 fatty acids, protein, vitamin D, B12,B3,B6 and mineral like 
potassium(Mel, 2024). The benefit is can improves eye health and this fish commonly cooked 
in Malaysia as “asam pedas” or just simply fried or grilled to making it healthy to any diet(Mel, 2024)
'''
st.markdown(multi)

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "SARIMA", "ARIMA", "LSTM"])


with tab1: 
  # Filter data for item_code = 55
  item_55_data = df[df['item_code'] == 55].copy()

  # Convert 'date' to datetime format
  item_55_data['date'] = pd.to_datetime(item_55_data['date'], format='%d-%b-%y')

  # Set 'date' as the index
  item_55_data.set_index('date', inplace=True)

  # Aggregating prices by date (average price per day)
  item_55_daily_prices = item_55_data.groupby('date')['price'].mean()

  # Display the processed data
  item_55_daily_prices.head(), item_55_daily_prices.info()

  st.title("Ikan Kembung Daily Average Prices")
  # Plot the time series data
  plt.figure(figsize=(12, 6))
  plt.plot(item_55_daily_prices, label='Daily Average Price', color='blue')
  plt.title('Daily Average Prices of Item 55', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Price (RM)', fontsize=12)
  plt.legend()
  plt.grid(True)
  plt.show()
  st.pyplot(plt.gcf())
  multi = '''From July to September 2023, the average daily price changed a lot, moving between 
RM13.50 and RM14.50 per kilogram. This shows that the market was not stable. In October 
2023, there was a small drop in prices and they stayed around RM13.00 for a short time before 
going up and down again. In November 2023, the price dropped sharply to about RM12.00, 
which was one of the lowest points of the year. This could have happened because of changes 
in the market or outside factors like USD prices or shifts in demand. From December 2023 to 
February 2024, the price started to recover, going back up to around RM13.50. During this 
74 
time, prices stayed the same and steady compared to the ups and downs seen in 2023. In March 
2024, there was a sudden and big drop with the price falling below RM5.00 for a short time. 
This sharp fall could be because of market change or some issue. Starting from April 2024, the 
price went back up and stayed around RM12.00 to RM13.00.However, there were still some 
changes, with prices sometimes reaching RM14.00 in June 2024. 
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

# Filter the data for a specific item, e.g., item_code 1
  item_code_to_check = 55.0
  filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

  st.title("Ikan Kembung Average Monthly Prices")

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
  multi = '''In July 2023, the price started high, averaging around RM14.00. This high price stayed 
almost the same until September 2023, with only small changes. Then in October 2023, the 
price dropped sharply to RM10.00, which was the lowest point of the year. After a sharp fall, 
the price bounced back quickly in November 2023, climbing to around RM14.00, showing a 
fast market recovery or adjustment. After reaching its highest point in November, the price 
started to drop slowly, falling to RM12.00 by March 2024. In April 2024, the price reached its 
lowest level since November 2023, staying close to RM11.00. This shows the downward trend 
from earlier months continued. From May 2024, the price began to stabilize and improve 
slightly, staying between RM11.50 and RM12.00 in June 2024. However, the recovery was 
small compared to the higher prices in mid-2023. 
'''
  st.markdown(multi)

  st.title("Diagnostic Analysis")

  multi = ''' The price trend for “ikan kembung” shows there several effect that causing the price 
increase and decrease. During the Northeast Monsoon (MTL) in late September 2023, rough 
seas reduces fish catches, as fisherman faced challenges going to sea due to string waves. This 
supply shortage persisted until December 2023, price stabilized due to favourable weather 
conditions that allowed sufficient fish supply despite the rainy season(Hidayatidayu Razali, 
2023). Additionally, the “Rahmah” program already started in January also played a role in 
stabilizing the supply by reducing market price and ensuring affordable access to fish also to 
help people who were affected by the flood that happened from December to January 2024. 
(Bernama, 2024). 
'''
  st.markdown(multi)
  multi = ''' Market Demand also influenced item prices significantly, especially during March 
2024, when Ramadhan began. The high Muslim population in Malaysia typically increases 
food demand during the fasting month. However, the availability of discounted goods under 
the Rahmah initiative helps stabilize prices despite higher consumption(Bernama, 2024). 
Similarly in April 2024, during the festive season of Eid al-Fitr, demand for food items 
increased but the government implemented a price control scheme for Indian mackerel to make 
sure the price is not increased spike (KPDNK, 2024). 
'''
  st.markdown(multi)

  multi = '''Economic factors such as reliance on imports and a weak currency also played a role in 
the price hike, In June 2023, prices spiked sharply as reduced local fish catches led to increase 
dependency on imports. The weak currency further worsens the cost of “ikan kembung”, 
contributing to the sharp increase in prices observed in the price(Hidayatidayu Razali, 2023). 
'''
  st.markdown(multi)
  multi = ''' Climate change also give an impact for “ikan kembung” prices. Reports show that 
climate- related changes, such as hot weather and sea storms, have led to 60% to 70% decrease 
in “ikan kembung” catches, resulting in a significant reduction in supply. This shows in June 
2023, when hot weather disrupted fishing activities, showing the price in the graph started  
higher(Hidayatidayu Razali, 2023). Also the  MTL  in late 2023 showing limited fishing 
activities due to rough seas, causing to supply shortages and make price increase(Zuliaty 
Zulkiffli, 2024). 
'''
  st.markdown(multi)

with tab2:
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd
  import numpy as np
  from statsmodels.tsa.stattools import adfuller
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  from sklearn.metrics import mean_squared_error, mean_absolute_error

  # Load and prepare the dataset
  try:
    data = pd.read_csv('/content/combined_filtered_allyears .csv')

    # Filter for item_code 55 and process dates
    item_55_data = data[data['item_code'] == 55].copy()
    item_55_data['date'] = pd.to_datetime(item_55_data['date'], format='%d-%b-%y')

    # Aggregate price by date and handle missing values
    item_55_aggregated = item_55_data.groupby('date')['price'].mean().reset_index()
    item_55_aggregated.set_index('date', inplace=True)
    item_55_aggregated = item_55_aggregated.asfreq('D')

    # Use interpolation for missing values
    item_55_aggregated['price'] = item_55_aggregated['price'].interpolate(method='time')

  except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise

  # Plot original time series
  plt.figure(figsize=(12, 6))
  plt.plot(item_55_aggregated.index, item_55_aggregated['price'], label="Observed Prices", color="blue")
  plt.title("Price Trend for Item Code 55")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  # Perform the Augmented Dickey-Fuller test for stationarity
  adf_result = adfuller(item_55_aggregated['price'])
  print("\nADF Test Results (Original Series):")
  print(f"ADF Statistic: {adf_result[0]:.4f}")
  print(f"p-value: {adf_result[1]:.4f}")
  print("Critical Values:")
  for key, value in adf_result[4].items():
    print(f"\t{key}: {value:.4f}")

  # Plot ACF and PACF
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  plot_acf(item_55_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
  plot_pacf(item_55_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
  plt.tight_layout()
  plt.show()

  # Fit SARIMA model with specified parameters (2,2,2)(0,1,1,12)
  print("\nFitting SARIMA(2,2,2)(0,1,1,12) model...")
  model = SARIMAX(
    item_55_aggregated['price'],
    order=(2, 2, 2),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
  )

  try:
    fitted_model = model.fit(disp=False)
    print("\nModel Summary:")
    print(fitted_model.summary())

    # Model diagnostics
    fitted_model.plot_diagnostics(figsize=(15, 8))
    plt.tight_layout()
    plt.show()

    # Generate forecast
    forecast_steps = 30
    forecast = fitted_model.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        start=item_55_aggregated.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps
    )

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(item_55_aggregated['price'], label="Observed", color="blue")
    plt.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="orange")
    plt.fill_between(
        forecast_index,
        forecast.conf_int().iloc[:, 0],
        forecast.conf_int().iloc[:, 1],
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title("SARIMA Model Forecast for Item Code 55")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error fitting model: {e}")
    raise
