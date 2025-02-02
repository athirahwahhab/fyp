import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)   

st.title("Ayam Standard")
multi = '''Ayam Standard refers to chicken that has undergone the standard process of slaughtering and cleaning, ensuring it is prepared according to hygiene and quality standards.
This preparation includes the whole chicken carcass and key parts such as the head, feet, liver, and gizzard, which remain intact and not removed. 
Under the Malaysian government's Maximum Price Control Scheme, the retail price for Ayam Standard was set at RM8.90 per kilogram in Malaysia starting as of June 2022(KPDNK, 2022).
This classification is to ensure that households have access to affordable chicken options while maintaining market stability. 
'''
st.markdown(multi)
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

st.title("Prediction Model ARIMA")

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
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(daily_avg['price'])
plt.title(f' Average Daily Price for Item {item_code}')
plt.show()
st.pyplot(plt.gcf())
# Test for stationarity
rolmean = daily_avg['price'].rolling(12).mean()
rolstd = daily_avg['price'].rolling(12).std()

plt.figure(figsize=(10,6))
plt.plot(daily_avg['price'], color='blue',label='Original')
plt.plot(rolmean, color='red', label='Rolling Mean')
plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.show()
st.pyplot(plt.gcf())
print("Results of Dickey-Fuller Test:")
adft = adfuller(daily_avg['price'].dropna(),autolag='AIC')
output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
    output['critical value (%s)'%key] =  values
print(output)

# Decompose the time series
clean_data = daily_avg['price'].dropna()
result = seasonal_decompose(clean_data, model='multiplicative', period=30)
fig = plt.figure(figsize=(16,9))
result.plot()
plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())
# Prepare data
df_log = np.log(daily_avg['price'].dropna())

# Build model with specified parameters using all data
# ARIMA(2,0,0)(1,1,1)[7]
model = SARIMAX(df_log,
                order=(2, 0, 0),
                seasonal_order=(1, 1, 1, 7))
fitted = model.fit()

print("\nARIMA Model Results:")
print(fitted.summary())

# Get fitted values for historical data
fitted_values = fitted.get_prediction(start=0)
fitted_mean = fitted_values.predicted_mean

# Calculate evaluation metrics on historical data
actual_values = df_log
predicted_values = fitted_mean

# Transform back to original scale for metrics
actual_orig = np.exp(actual_values)
predicted_orig = np.exp(predicted_values)

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
forecast_results = fitted.get_forecast(forecast_period)
fc = forecast_results.predicted_mean
conf_int = forecast_results.conf_int()

# Generate future dates starting from the end of actual data
last_date = df_log.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                           periods=forecast_period,
                           freq='D')

# Transform back to original scale
fc_series = np.exp(pd.Series(fc, index=future_dates))
lower_series = np.exp(pd.Series(conf_int.iloc[:, 0], index=future_dates))
upper_series = np.exp(pd.Series(conf_int.iloc[:, 1], index=future_dates))
historical_data = np.exp(df_log)
fitted_historical = np.exp(fitted_mean)

plt.figure(figsize=(12,6), dpi=100)
plt.plot(historical_data, label='Actual Data', alpha=0.8)
plt.plot(fc_series, color='yellow', label='Predicted Price')
plt.fill_between(future_dates, lower_series, upper_series, color='yellow', alpha=.1, label='Confidence Interval')


# Load and prepare data
item_code = 1  # Specify which item to analyze

df = pd.read_csv('/content/combined_filtered_allyears .csv', parse_dates=['date'])
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
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(daily_avg['price'])
plt.title(f' Average Daily Price for Item {item_code}')
plt.show()
st.pyplot(plt.gcf())
# Test for stationarity
rolmean = daily_avg['price'].rolling(12).mean()
rolstd = daily_avg['price'].rolling(12).std()

plt.figure(figsize=(10,6))
plt.plot(daily_avg['price'], color='blue',label='Original')
plt.plot(rolmean, color='red', label='Rolling Mean')
plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.show()
st.pyplot(plt.gcf())
print("Results of Dickey-Fuller Test:")
adft = adfuller(daily_avg['price'].dropna(),autolag='AIC')
output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
    output['critical value (%s)'%key] =  values
print(output)

# Decompose the time series
clean_data = daily_avg['price'].dropna()
result = seasonal_decompose(clean_data, model='multiplicative', period=30)
fig = plt.figure(figsize=(16,9))
result.plot()
plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())
# Prepare data
df_log = np.log(daily_avg['price'].dropna())

# Build model with specified parameters using all data
# ARIMA(2,0,0)(1,1,1)[7]
model = SARIMAX(df_log,
                order=(2, 0, 0),
                seasonal_order=(1, 1, 1, 7))
fitted = model.fit()

print("\nARIMA Model Results:")
print(fitted.summary())

# Get fitted values for historical data
fitted_values = fitted.get_prediction(start=0)
fitted_mean = fitted_values.predicted_mean

# Calculate evaluation metrics on historical data
actual_values = df_log
predicted_values = fitted_mean

# Transform back to original scale for metrics
actual_orig = np.exp(actual_values)
predicted_orig = np.exp(predicted_values)

mse = mean_squared_error(actual_orig, predicted_orig)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_orig, predicted_orig)
mape = np.mean(np.abs(actual_orig - predicted_orig) / np.abs(actual_orig)) * 100

print('\nModel Performance Metrics (Historical Data):')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}%')

# Forecast 90 days into the future
forecast_period = 90  # 3 months
forecast_results = fitted.get_forecast(steps=forecast_period) # Added steps parameter
fc = forecast_results.predicted_mean
conf_int = forecast_results.conf_int()

# Generate future dates starting from the end of actual data
last_date = df_log.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                           periods=forecast_period,
                           freq='D')

# Transform back to original scale
fc_series = np.exp(pd.Series(fc, index=future_dates))
lower_series = np.exp(pd.Series(conf_int.iloc[:, 0], index=future_dates))
upper_series = np.exp(pd.Series(conf_int.iloc[:, 1], index=future_dates))
historical_data = np.exp(df_log)
fitted_historical = np.exp(fitted_mean)

plt.figure(figsize=(12,6), dpi=100)
plt.plot(historical_data, label='Actual Data', alpha=0.8)
plt.plot(fc_series, color='yellow', label='Predicted Price')
plt.fill_between(future_dates, lower_series, upper_series, color='red', alpha=.1, label='Confidence Interval')


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

#visualize the graph only show predicted price and confidence interval

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6), dpi=100)
plt.plot(fc_series, color='orange', label='Predicted Price')
plt.fill_between(future_dates, lower_series, upper_series, color='orange', alpha=.1, label='Confidence Interval')
plt.axvline(x=test_data.index[-1], color='r', linestyle='--', alpha=0.5, label='End of Historical Data')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.show()
st.pyplot(plt.gcf())
