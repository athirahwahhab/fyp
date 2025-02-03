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

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "SARIMA", "ARIMA", "LSTM"])


with tab1: 
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
    data = pd.read_csv('https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv')
    
    # Filter for item_code 69 and process dates
    item_69_data = data[data['item_code'] == 69].copy()
    item_69_data['date'] = pd.to_datetime(item_69_data['date'], format='%d-%b-%y')

    # Aggregate price by date and handle missing values
    item_69_aggregated = item_69_data.groupby('date')['price'].mean().reset_index()
    item_69_aggregated.set_index('date', inplace=True)
    item_69_aggregated = item_69_aggregated.asfreq('D')

    # Use interpolation for missing values
    item_69_aggregated['price'] = item_69_aggregated['price'].interpolate(method='time')
  
  except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise

  # Plot original time series
  plt.figure(figsize=(12, 6))
  plt.plot(item_69_aggregated.index, item_69_aggregated['price'], label="Observed Prices", color="blue")
  plt.title("Price Trend for Item Code 69")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  # Perform the Augmented Dickey-Fuller test for stationarity
  adf_result = adfuller(item_69_aggregated['price'])
  print("\nADF Test Results (Original Series):")
  print(f"ADF Statistic: {adf_result[0]:.4f}")
  print(f"p-value: {adf_result[1]:.4f}")
  print("Critical Values:")

  for key, value in adf_result[4].items():
    print(f"\t{key}: {value:.4f}")

  # Plot ACF and PACF
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  plot_acf(item_69_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
  plot_pacf(item_69_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
  plt.tight_layout()
  plt.show()

  # Fit SARIMA model with specified parameters (2,2,2)(0,1,1,12)
  print("\nFitting SARIMA(2,2,2)(0,1,1,12) model...")
  model = SARIMAX( item_69_aggregated['price'], order=(2, 2, 2), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False )

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
        start=item_69_aggregated.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps
    )

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(item_69_aggregated['price'], label="Observed", color="blue")
    plt.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="orange")
    plt.fill_between(
        forecast_index,
        forecast.conf_int().iloc[:, 0],
        forecast.conf_int().iloc[:, 1],
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title("SARIMA Model Forecast for Item Code 69")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())

  except Exception as e:
    print(f"Error fitting model: {e}")
    raise
    
with tab3:
  import warnings
  warnings.filterwarnings('ignore')
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from statsmodels.tsa.stattools import adfuller
  from statsmodels.tsa.seasonal import seasonal_decompose
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from sklearn.metrics import mean_squared_error, mean_absolute_error
  import math

  # Load and prepare data
  item_code = 69  # Specify which item to analyze
  df = pd.read_csv('https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv', parse_dates=['date'])
  item_data = df[df['item_code'] == item_code].copy()

  # Calculate daily average
  daily_avg = item_data.groupby('date')['price'].mean().reset_index()

  # Handle missing values using interpolation
  daily_avg.set_index('date', inplace=True)
  daily_avg.sort_index(inplace=True)
  idx = pd.date_range(daily_avg.index.min(), daily_avg.index.max())
  daily_avg = daily_avg.reindex(idx)
  daily_avg['price'] = daily_avg['price'].interpolate(method='linear')

  # Plot initial price history
  plt.figure(figsize=(10, 6))
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.plot(daily_avg['price'])
  plt.title(f'Average Daily Price for Item {item_code}')
  plt.show()

  # Test for stationarity
  rolmean = daily_avg['price'].rolling(12).mean()
  rolstd = daily_avg['price'].rolling(12).std()

  plt.figure(figsize=(10, 6))
  plt.plot(daily_avg['price'], color='blue', label='Original')
  plt.plot(rolmean, color='red', label='Rolling Mean')
  plt.plot(rolstd, color='black', label='Rolling Std')
  plt.legend(loc='best')
  plt.title('Rolling Mean and Standard Deviation')
  plt.show()

  print("Results of Dickey-Fuller Test:")
  adft = adfuller(daily_avg['price'].dropna(), autolag='AIC')
  output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
  for key, values in adft[4].items():
      output[f'critical value ({key})'] = values
  print(output)

  # Decompose the time series
  clean_data = daily_avg['price'].dropna()
  result = seasonal_decompose(clean_data, model='multiplicative', period=30)
  fig = plt.figure(figsize=(16, 9))
  result.plot()
  plt.tight_layout()
  plt.show()

  # Prepare data
  df_log = np.log(daily_avg['price'].dropna())

  # Build model with specified parameters using all data
  model = SARIMAX(df_log, order=(2, 0, 0), seasonal_order=(1, 1, 1, 7))
  fitted = model.fit()

  print("\nARIMA Model Results:")
  print(fitted.summary())

  # Get fitted values for historical data
  fitted_values = fitted.get_prediction(start=0)
  fitted_mean = fitted_values.predicted_mean

  # Transform back to original scale for metrics
  actual_orig = np.exp(df_log)
  predicted_orig = np.exp(fitted_mean)

  # Calculate evaluation metrics
  mse = mean_squared_error(actual_orig, predicted_orig)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(actual_orig, predicted_orig)
  mape = np.mean(np.abs(actual_orig - predicted_orig) / np.abs(actual_orig)) * 100

  print('\nModel Performance Metrics (Historical Data):')
  print(f'RMSE: {rmse:.4f}')
  print(f'MAE: {mae:.4f}')
  print(f'MAPE: {mape:.4f}')

  # Forecast 90 days into the future
  forecast_period = 90  # 3 months
  forecast_results = fitted.get_forecast(steps=forecast_period)
  fc = forecast_results.predicted_mean
  conf_int = forecast_results.conf_int()

  # Generate future dates starting from the end of actual data
  last_date = df_log.index[-1]
  future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                               periods=forecast_period, freq='D')

  # Transform back to original scale
  fc_series = np.exp(pd.Series(fc, index=future_dates))
  lower_series = np.exp(pd.Series(conf_int.iloc[:, 0], index=future_dates))
  upper_series = np.exp(pd.Series(conf_int.iloc[:, 1], index=future_dates))
  historical_data = np.exp(df_log)
  fitted_historical = np.exp(fitted_mean)

  plt.figure(figsize=(12, 6), dpi=100)
  plt.plot(historical_data, label='Actual Data', alpha=0.8)
  plt.plot(fc_series, color='orange', label='Predicted Price')
  plt.fill_between(future_dates, lower_series, upper_series, color='orange', alpha=0.1, label='Confidence Interval')
  plt.title(f'Price Prediction Item Code {item_code}')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend(loc='upper left', fontsize=8)
  plt.grid(True, alpha=0.3)
  plt.show()
  st.pyplot(plt.gcf())
