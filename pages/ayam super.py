import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

st.title("Ayam Super")

multi = ''' Ayam Super is a whole chicken sold without the head, feet, liver, and gizzard. This makes it more processed and convenient for cooking, catering to consumers who prefer less preparation work. 
As of June 2022, the retail price for Ayam Super was capped at RM9.90 per kilogram in Peninsular Malaysia under the same government price control initiative. 
Ayam Super is slightly more expensive than Ayam Standard due to the additional cleaning and processing involved, making it a preferred choice for those seeking convenience.
'''
st.markdown(multi)

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "SARIMA", "ARIMA", "LSTM"])


with tab1: 
  item_2_data = df[df['item_code'] == 2 ].copy()

# Convert 'date' to datetime format
  item_2_data['date'] = pd.to_datetime(item_2_data['date'], format='%d-%b-%y')

# Set 'date' as the index
  item_2_data.set_index('date', inplace=True)

# Aggregating prices by date (average price per day)
  item_2_daily_prices = item_2_data.groupby('date')['price'].mean()

# Display the processed data
  item_2_daily_prices.head(), item_2_daily_prices.info()
  
  st.title("Ayam Super Average Daily Prices ")
  # Plot the time series data
  plt.figure(figsize=(12, 6))
  plt.plot(item_2_daily_prices, label='Daily Average Price', color='blue')
  plt.title('Daily Average Prices of Item 2', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Price (RM)', fontsize=12)
  plt.legend()
  plt.grid(True)
  plt.show()
  st.pyplot(plt.gcf())

  multi = '''At the beginning of July 2023, the price start was quite high, around RM10.00 per kilogram. 
Throughout the month, the price stayed stable at around RM 10.00 before it went down. 
In August 2023, there was a sudden drop in price falling about RM 8.30 which was the lowest point during this time.
From September to November 2023, prices slowly began to rise again, reaching about RM 9.50 by November. 
Between January and March 2024, prices went up and down a lot, moving between RM8.00 and RM 9.50 without a clear direction. 
This time was full of sudden and frequent price changes, possibly because of issues with supply and demand or other factors affecting the market. 
The most unstable time was between March and May 2024. Prices kept going up above RM 9.50 and then dropping below RM8.50. 
From the end of May 2024, prices started going up again and reached their highest point at around RM 10.00 in June 2024. 
This was the highest price since July 2023. In July 2024, prices kept up and down between RM9.00 and RM9.50. 
This shows things were a bit more stable but still changing a lot.
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

  st.title("Ayam Super Average Monthly Prices")
  # Filter the data for a specific item, e.g., item_code 2
  item_code_to_check = 2
  filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

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
  multi = '''In July 2023, the price was approximately RM 9.75 per kilogram, which is one of the highest monthly averages during this time.
This shows a strong market value at the beginning of the analysis. 
Next, in August 2023 the price dropped sharply to the lowest average of about RM 8.25 per kilogram. 
This was the biggest drop during the entire period. After the big drop in August, the price rising back in September 2023 to around RM9.25.
From October to December 2023, prices stayed steady at around RM9.00, with very little change. 
This shows the market was balanced in the last few months of the year. 
From January to April 2024, the price stayed mostly stable at around RM9.00. 
However, there were small drops in February and April bringing the average price down to RM8.75 during those months. 
This time showed some ups and downs but no big problems. In May 2024, prices began to go up again, reaching about RM9.25 by June 2024. 
Overall the prices went through quick drops and slow changes, showing shifts in the market.
'''
  st.markdown(multi)

  st.title("Diagnostic Analysis")

  multi = ''' The factor influencing chicken type super prices from June 2023 to June 2024. 
Ayam Super is  more expensive than Ayam Standard due to the additional cleaning and processing involved, making it a preferred choice for those seeking convenience. 
Same as chicken standard, supply chain is one of the factors in price trend. 
The government give an extension of subsidies and price control for chicken starting July 2023 (Yusof, 2023) helped reduce production costs leading to a price drop from RM9.75 to RM8.25 between June and August 2023. 
This decision was made by the Malaysia government to ensure consumer welfare and maintain stability in the poultry industry, as reported in the proposal by the Ministry of Agriculture and Food Security (KPKM) and the Ministry of Domestic Trade and Cost of Living (KPDN). 
Monitoring by KPDN shows a price decline in August due to effective enforcement and a reduction in inflationary pressures. 
From September to October 2023, the price went up to RM9.50 and then settled at RM9.00. 
This change probably happened because producers adjusted their output after having too much supply earlier in the year, and because the costs of feed and transportation went up. 
The government also set a maximum price for chicken at RM11.40, which helped keep prices stable from November 2023 to February 2024, even though they stopped giving subsidies in November 2023. 
The Department of Veterinary Services (DVS) said there was a steady supply of chicken during this time(Rosli, 2024).

'''
  st.markdown(multi)
  multi = ''' Economic factors also have an impact on the chicken prices. 
Inflation in Malaysia caused prices to go up and down, especially from August 2023 onwards. 
The rising costs of chicken feed and transportation led to higher prices in the second half of 2023. 
In March 2024, prices went up slightly to RM9.00, which happened around the start of Ramadhan. 
During Ramadan, demand for chicken usually increases because people need it for sahur and iftar as mentioned by Bank Negara Malaysia.  
From April to June 2024, prices went up and down. There was a small drop in April, but by June, prices rose to RM9.25. 
This increase was likely due to higher costs for transportation, feed, and labour, made worse by inflation. 
It was also influenced by preparations for Hari Raya Aidiladha, which took place at the end of June(Bank Negara, 2024).
'''
  st.markdown(multi)

  multi = ''' From economic perspective, inflation played a crucial role especially from April to June 2024. 
As inflation rise from 1.8% to 2% the cost of chicken also increased. 
Higher feed costs, plus with rising transportation expenses made chicken more expensive during that period(Malaysia Inflation Rate, 2020).
'''
  st.markdown(multi)
  multi = ''' Market Fundamentals and demand also a key factor in determining price changes. 
From June to August 2023, prices fell sharply because of government support and price limits. 
However, prices started to recover from September to October 2023 as production levels changed and demand grew due to the season. 
The Ramadan period in March 2024 also caused prices to rise because more chicken was needed for meals during sahur and iftar by the Muslim community(Danish, 2024). 
Similarly, prices increased before Hari Raya as people bought more food for the celebrations.
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

    # Filter for item_code 2 and process dates
      item_2_data = data[data['item_code'] == 2].copy()
      item_2_data['date'] = pd.to_datetime(item_2_data['date'], format='%d-%b-%y')

    # Aggregate price by date and handle missing values
      item_2_aggregated = item_2_data.groupby('date')['price'].mean().reset_index()
      item_2_aggregated.set_index('date', inplace=True)
      item_2_aggregated = item_2_aggregated.asfreq('D')

    # Use interpolation for missing values instead of forward fill
      item_2_aggregated['price'] = item_2_aggregated['price'].interpolate(method='time')
    
  except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise

  # Plot original time series
  plt.figure(figsize=(12, 6))
  plt.plot(item_2_aggregated.index, item_2_aggregated['price'], label="Observed Prices", color="blue")
  plt.title("Price Trend for Item Code 2")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  # Perform the Augmented Dickey-Fuller test for stationarity
  adf_result = adfuller(item_2_aggregated['price'])
  print("\nADF Test Results (Original Series):")
  print(f"ADF Statistic: {adf_result[0]:.4f}")
  print(f"p-value: {adf_result[1]:.4f}")
  print("Critical Values:")
  for key, value in adf_result[4].items():
    print(f"\t{key}: {value:.4f}")
  # Plot ACF and PACF
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  plot_acf(item_2_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
  plot_pacf(item_2_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
  plt.tight_layout()
  plt.show()
  # Fit SARIMA model with specified parameters (2,2,2)(0,1,1,12)
  print("\nFitting SARIMA(2,2,2)(0,1,1,12) model...")
  model = SARIMAX(
    item_2_aggregated['price'],
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
      start=item_2_aggregated.index[-1] + pd.Timedelta(days=1),
      periods=forecast_steps
    )

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(item_2_aggregated['price'], label="Observed", color="blue")
    plt.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="orange")
    plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color="orange", alpha=0.2, label="95% Confidence Interval")
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
  item_code = 2  # Specify which item to analyze
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
