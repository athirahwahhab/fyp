import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Ayam Standard")
multi = '''Ayam Standard refers to chicken that has undergone the standard process of slaughtering and cleaning, ensuring it is prepared according to hygiene and quality standards.
This preparation includes the whole chicken carcass and key parts such as the head, feet, liver, and gizzard, which remain intact and not removed. 
Under the Malaysian government's Maximum Price Control Scheme, the retail price for Ayam Standard was set at RM8.90 per kilogram in Malaysia starting as of June 2022(KPDNK, 2022).
This classification is to ensure that households have access to affordable chicken options while maintaining market stability. 
'''
st.markdown(multi)

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "LSTM", "SARIMA", "ARIMA"])
# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   
with tab1:
  # Filter data for item_code = 1
  item_1_data = df[df['item_code'] == 1].copy()
  # Convert 'date' to datetime format
  item_1_data['date'] = pd.to_datetime(item_1_data['date'], format='%d-%b-%y')
  # Set 'date' as the index
  item_1_data.set_index('date', inplace=True)

  # Aggregating prices by date (average price per day)
  item_1_daily_prices = item_1_data.groupby('date')['price'].mean()

  # Display the processed data
  item_1_daily_prices.head(), item_1_daily_prices.info()

  st.title("Ayam Standard Average Daily Prices ")
  # Plot the time series data
  plt.figure(figsize=(12, 6))
  plt.plot(item_1_daily_prices, label='Daily Average Price', color='blue')
  plt.title('Daily Average Prices of Item 1', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Price (RM)', fontsize=12)
  plt.legend()
  plt.grid(True)
  plt.show()
  # Display the plot in Streamlit
  st.pyplot(plt.gcf())

  multi = '''The graphs show at the beginning of the times at July 2023, the price was consistently high, around RM9.30 per kilogram. 
But by the end of July, it started to drop showing the beginning of a dropping price. 
In August 2023, the price fell hardly reaching the lowest point at about RM 8.00 per kilogram. 
After that, the price started to level out, moving between RM8.10 and RM 8.50 per kilogram. 
Starting in November 2023, the price slowly went up reaching around RM 9.00 by January 2024. 
Between February and April 2024, chicken prices went up and down a lot, with many sudden increases and decreases. 
Prices moved between RM 8.30 and RM 9.30 per kilogram, showing that the market was unstable. 
And lastly, in May 2024, prices began to go up steadily and stayed around RM9.20 to RM 9.30 by July 2024.
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
  item_code_to_check = 1
  filtered_item_data = average_price_per_item_monthly[average_price_per_item_monthly['item_code'] == item_code_to_check]

  st.title("Ayam Standard Average Monthly Prices")

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
  # Display the plot in Streamlit
  st.pyplot(plt.gcf())

  multi = '''The price of “Ayam standard” in Kelantan went up and down a lot from June 2023 to June 2024. The highest price was around RM9.30 per 
kilogram(kg) in June 2023, but by August 2023 it dropped and it became the lowest price that it was about RM 8.10 in August 2023. 
After that, the price started to slowly go up again. From September to December 2023,
the price stayed mostly steady, moving between RM8.20 and RM 8.60. As it entered 2024, the price changes a little.
In January it went up a bit but then dropped a bit in February. By March 2024 the price increased again, showing a steady improvement. 
Starting in April 2024, the price has been rising again, reaching RM 9.30 in June 2024. 
This price matched the highest price at the beginning of the recorded price in June 2023.
'''
  st.markdown(multi)

  st.title("Diagnostic Analysis")

  multi = ''' The factors that contributes chicken prices from June 2023 to June 2024:
Supply chain really played a significant role in the price ups and downs. In mid-2023, 
government subsidies and price controls helped reduce production costs leading to a price drop, 
especially when the “Jualan Rahmah” program offered discounted chicken to the public.
However, in the later months, severe floods in Kelantan caused by the monsson season disrupted transportation and the overall supply chain,
making it difficult to deliver goods to markets. This make the price higher due to logistical challenges and road closures as reported in The Star News about Flood Woes Worsen in Kelantan, (2023).  
Additionally, inflation from April to June 2024 worse the supply chain issues, including higher transportation costs and feed prices, making it even harder to maintain steady chicken prices shown (Consumer Prices | OpenDOSM, 2025).
'''
  st.markdown(multi)
  multi = ''' Climate change also had a impact in the price trend particularly through the monsoon season. 
The end-of-year flooding in Kelantan disrupted the chicken supply chain, affecting market operations and make prices increase. 
The seasonal flooding typically hinders production and distribution, further complicating the supply of chicken(NADMA, 2023).
'''
  st.markdown(multi)

  multi = ''' From economic perspective, inflation played a crucial role especially from April to June 2024. 
As inflation rise from 1.8% to 2% the cost of chicken also increased. 
Higher feed costs, plus with rising transportation expenses made chicken more expensive during that period(Malaysia Inflation Rate, 2020).
'''
  st.markdown(multi)
  multi = ''' Lastly, market fundamentals and demand were also influencing price trends. 
In  the middle of 2023, a drop-in chicken prices were because oversupply and decreased consumer demand(Jamaludin et al., 2023). 
However, as the Ramadhan season approached in early 2024, there was a surge in demand for chicken particularly for iftar and sahur meals, which contributed to the gradual price increase (Bernama, 2024c). 
These market shifts, combined with the factors mentioned, shaped the overall price trend through the year.
'''
  st.markdown(multi)

with tab2:
  # Importing necessary libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.metrics import mean_squared_error
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, LSTM
  # Load the dataset
  df = pd.read_csv('https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_filtered_allyears%20.csv')

  # Filter data for item_code = 1
  item_1_data = df[df['item_code'] == 1].copy()

  # Convert 'date' to datetime format
  item_1_data['date'] = pd.to_datetime(item_1_data['date'], format='%d-%b-%y')

  # Set 'date' as the index
  item_1_data.set_index('date', inplace=True)

  # Group by date and calculate the average price
  item_1_daily_prices = item_1_data.groupby('date')['price'].mean()

  # Initial data visualization
  plt.figure(figsize=(10, 6))
  plt.plot(item_1_daily_prices, label='Daily Average Price', color='blue', marker='o')
  plt.title('Daily Average Prices of Item 1', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Price (RM)', fontsize=12)
  plt.legend()
  plt.grid(True)
  plt.show()
  # Prepare data for the LSTM model
  dataset = item_1_daily_prices.values.reshape(-1, 1)

  # Normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # Split into train and test sets
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

  # Convert an array of values into a dataset matrix
  def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

  # Reshape into X=t and Y=t+1
  look_back = 1
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)

  # Reshape input to be [samples, time steps, features]
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  # Create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

  # Make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # Invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])

  # Calculate root mean squared error
  trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
  print('Test Score: %.2f RMSE' % (testScore))

  # Create figure for final plot
  plt.figure(figsize=(10, 6))

  # Plot actual data
  actual_dates = item_1_daily_prices.index
  actual_values = scaler.inverse_transform(dataset)
  plt.plot(actual_dates, actual_values, label='Actual Data', color='blue')

  # Plot training predictions
  # Adjust indices to match the correct timeframe for training predictions
  train_dates = actual_dates[look_back:len(trainPredict) + look_back]
  plt.plot(train_dates, trainPredict, label='Train Predictions', color='green')

  # Plot test predictions
  # Carefully align test prediction dates
  test_start_idx = len(trainPredict) + look_back
  test_dates = actual_dates[test_start_idx:test_start_idx + len(testPredict)]
  plt.plot(test_dates, testPredict, label='Test Predictions', color='red')

  # Add labels and legend
  plt.title('LSTM Model Predictions vs Actual Data', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Price (RM)', fontsize=12)
  plt.xticks(rotation=45)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  st.pyplot(plt.gcf())
  def generate_future_predictions(model, last_sequence, n_future_days, scaler):

    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future_days):
        # Reshape the sequence for prediction
        current_sequence_reshaped = current_sequence.reshape((1, 1, 1))

        # Get the next predicted value
        next_pred = model.predict(current_sequence_reshaped, verbose=0)

        # Store the prediction
        future_predictions.append(next_pred[0, 0])

        # Update the sequence with the new prediction
        current_sequence = np.array([next_pred[0, 0]])

    # Inverse transform the predictions to get actual prices
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions

    # Get the last known sequence (using the last value from our dataset)
    last_known_seq = dataset[-1:]

    # Generate future dates
    last_date = item_1_daily_prices.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq='D')

    # Generate predictions for next 30 days
    n_future_days = 90
    future_predictions = generate_future_predictions(model, last_known_seq, n_future_days, scaler)

    # Create final visualization including future predictions
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(actual_dates, actual_values, label='Historical Data', color='blue')
    plt.plot(train_dates, trainPredict, label='Training Predictions', color='green')
    plt.plot(test_dates, testPredict, label='Test Predictions', color='red')
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='purple', linestyle='--')
    # Add confidence intervals for future predictions (simple approach)
    future_std = np.std(actual_values[-90:])  # Using last 30 days as reference
    plt.fill_between(future_dates, future_predictions.flatten() - future_std, future_predictions.flatten() + future_std, color='purple', alpha=0.2, label='Prediction Interval')
    plt.title('LSTM Model Predictions Including Future Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (RM)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())

with tab3:
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
      
  # Filter for item_code 1 and process dates
      item_1_data = data[data['item_code'] == 1].copy()
      item_1_data['date'] = pd.to_datetime(item_1_data['date'], format='%d-%b-%y')

    # Aggregate price by date and handle missing values
      item_1_aggregated = item_1_data.groupby('date')['price'].mean().reset_index()
      item_1_aggregated.set_index('date', inplace=True)
      item_1_aggregated = item_1_aggregated.asfreq('D')

    # Use interpolation for missing values
      item_1_aggregated['price'] = item_1_aggregated['price'].interpolate(method='time')
    except Exception as e:
      print(f"Error loading or processing data: {e}")
      raise
    # Plot original time series
    plt.figure(figsize=(12, 6))
    plt.plot(item_1_aggregated.index, item_1_aggregated['price'], label="Observed Prices", color="blue")
    plt.title("Price Trend for Item Code 1")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Perform the Augmented Dickey-Fuller test for stationarity
    adf_result = adfuller(item_1_aggregated['price'])
    print("\nADF Test Results (Original Series):")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
  
    for key, value in adf_result[4].items():
      print(f"\t{key}: {value:.4f}")
     # Plot ACF and PACF
      fig, axes = plt.subplots(1, 2, figsize=(15, 5))
      plot_acf(item_1_aggregated['price'], lags=40, ax=axes[0], title="ACF of Series")
      plot_pacf(item_1_aggregated['price'], lags=40, ax=axes[1], title="PACF of Series")
      plt.tight_layout()
      plt.show()

    # Fit SARIMA model with parameter
    model = SARIMAX(
      item_1_aggregated['price'],
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
      start=item_1_aggregated.index[-1] + pd.Timedelta(days=1),
      periods=forecast_steps
      )
# Plot forecast
      plt.figure(figsize=(12, 6))
      plt.plot(item_1_aggregated['price'], label="Observed", color="blue")
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
      # Print model metrics
      print("\nModel Metrics:")
      print(f"AIC: {fitted_model.aic:.2f}")
      print(f"BIC: {fitted_model.bic:.2f}")
    except Exception as e:
        print(f"Error fitting model: {e}")
        raise
      
with tab4:
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

  #Load and prepare data
  item_code = 1  # Specify which item to analyze
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
  # Print future predictions
  print("\nFuture price predictions:")
  print(fc_series)
