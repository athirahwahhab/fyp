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

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "SARIMA", "ARIMA", "LSTM"])

with tab1: 
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
      data = pd.read_csv('https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv')

      # Filter for item_code 1556 and process dates
      item_1556_data = data[data['item_code'] == 1556].copy()
      item_1556_data['date'] = pd.to_datetime(item_1556_data['date'], format='%d-%b-%y')

      # Aggregate price by date and handle missing values
      item_1556_aggregated = item_1556_data.groupby('date')['price'].mean().reset_index()
      item_1556_aggregated.set_index('date', inplace=True)
      item_1556_aggregated = item_1556_aggregated.asfreq('D')

      # Use interpolation for missing values
      item_1556_aggregated['price'] = item_1556_aggregated['price'].interpolate(method='time')

  except Exception as e:
      print(f"Error loading or processing data: {e}")
      raise

  # Plot original time series
  plt.figure(figsize=(12, 6))
  plt.plot(item_1556_aggregated.index, item_1556_aggregated['price'], label="Observed Prices", color="blue")
  plt.title("Price Trend for Item Code 1556")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

# Perform the Augmented Dickey-Fuller test for stationarity
  adf_result = adfuller(item_1556_aggregated['price'])
  print("\nADF Test Results (Original Series):")
  print(f"ADF Statistic: {adf_result[0]:.4f}")
  print(f"p-value: {adf_result[1]:.4f}")
  print("Critical Values:")
  for key, value in adf_result[4].items():
      print(f"\t{key}: {value:.4f}")

# Plot ACF and PACF
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  plot_acf(item_1556_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
  plot_pacf(item_1556_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
  plt.tight_layout()
  plt.show()

  # Fit SARIMA model with specified parameters (2,2,2)(0,1,1,12)
  print("\nFitting SARIMA model...")
  model = SARIMAX(
      item_1556_aggregated['price'],
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
        start=item_1556_aggregated.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps
    )

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(item_1556_aggregated['price'], label="Observed", color="blue")
    plt.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="orange")
    plt.fill_between(
        forecast_index,
        forecast.conf_int().iloc[:, 0],
        forecast.conf_int().iloc[:, 1],
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title("SARIMA Model Forecast")
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
  item_code = 1556  # Specify which item to analyze
  df = pd.read_csv('https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv', parse_dates=['date'])
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
