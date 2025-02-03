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

tab1, tab2, tab3, tab4 = st.tabs(["Price Trend ", "LSTM", "SARIMA", "ARIMA"])

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
  # Filter data for item_code = 2
item_2_data = df[df['item_code'] == 2].copy()

# Convert 'date' to datetime format
item_2_data['date'] = pd.to_datetime(item_2_data['date'], format='%d-%b-%y')

# Set 'date' as the index
item_2_data.set_index('date', inplace=True)

# Group by date and calculate the average price
item_2_daily_prices = item_2_data.groupby('date')['price'].mean()

# Initial data visualization
plt.figure(figsize=(10, 6))
plt.plot(item_2_daily_prices, label='Daily Average Price', color='blue', marker='o')
plt.title('Daily Average Prices of Item 2 ', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for the LSTM model
dataset = item_2_daily_prices.values.reshape(-1, 1)

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
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

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
actual_dates = item_2_daily_prices.index
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
last_date = item_2_daily_prices.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                           periods=90,
                           freq='D')

# Generate predictions for next 30 days
n_future_days = 90
future_predictions = generate_future_predictions(model,
                                              last_known_seq,
                                              n_future_days,
                                              scaler)

# Create final visualization including future predictions
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(actual_dates, actual_values, label='Historical Data', color='blue')
plt.plot(train_dates, trainPredict, label='Training Predictions', color='green')
plt.plot(test_dates, testPredict, label='Test Predictions', color='red')

# Plot future predictions
plt.plot(future_dates, future_predictions,
         label='Future Predictions',
         color='purple',
         linestyle='--')

# Add confidence intervals for future predictions (simple approach)
future_std = np.std(actual_values[-90:])  # Using last 30 days as reference
plt.fill_between(future_dates,
                future_predictions.flatten() - future_std,
                future_predictions.flatten() + future_std,
                color='purple',
                alpha=0.2,
                label='Prediction Interval')

plt.title('LSTM Model Predictions Including Future Forecast', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (RM)', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())
# Print future predictions with dates
future_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions.flatten(),
    'Lower_Bound': future_predictions.flatten() - future_std,
    'Upper_Bound': future_predictions.flatten() + future_std
})

print("\nFuture Price Predictions:")
print(future_forecast.to_string(index=False))

  

