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

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "SARIMA", "ARIMA", "LSTM"])

with tab1:
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
  
      # Filter for item_code 1111 and process dates
      item_1111_data = data[data['item_code'] == 1111].copy()
      item_1111_data['date'] = pd.to_datetime(item_1111_data['date'], format='%d-%b-%y')
  
      # Aggregate price by date and handle missing values
      item_1111_aggregated = item_1111_data.groupby('date')['price'].mean().reset_index()
      item_1111_aggregated.set_index('date', inplace=True)
      item_1111_aggregated = item_1111_aggregated.asfreq('D')
  
      # Use interpolation for missing values
      item_1111_aggregated['price'] = item_1111_aggregated['price'].interpolate(method='time')
  
  except Exception as e:
      print(f"Error loading or processing data: {e}")
      raise
  
  # Plot original time series
  plt.figure(figsize=(12, 6))
  plt.plot(item_1111_aggregated.index, item_1111_aggregated['price'], label="Observed Prices", color="blue")
  plt.title("Price Trend for Item Code 1111")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()
  
  # Perform the Augmented Dickey-Fuller test for stationarity
  adf_result = adfuller(item_1111_aggregated['price'])
  print("\nADF Test Results (Original Series):")
  print(f"ADF Statistic: {adf_result[0]:.4f}")
  print(f"p-value: {adf_result[1]:.4f}")
  print("Critical Values:")
  for key, value in adf_result[4].items():
      print(f"\t{key}: {value:.4f}")
  
  # Plot ACF and PACF
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  plot_acf(item_1111_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
  plot_pacf(item_1111_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
  plt.tight_layout()
  plt.show()
  
  # Fit SARIMA model with parameter
  model = SARIMAX(
      item_1111_aggregated['price'],
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
          start=item_1111_aggregated.index[-1] + pd.Timedelta(days=1),
          periods=forecast_steps
      )
  
      # Plot forecast
      plt.figure(figsize=(12, 6))
      plt.plot(item_1111_aggregated['price'], label="Observed", color="blue")
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
  item_code = 1111  # Specify which item to analyze
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
