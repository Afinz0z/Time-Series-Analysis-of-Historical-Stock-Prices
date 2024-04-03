import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("your_dataset.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')  # Updated format
data.set_index('Date', inplace=True)

print(data.head())

print(data.describe())

print(data.isnull().sum())

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(data.columns)
plt.show()

decomposition = seasonal_decompose(data['AMZN'], model='additive', period=30)  # Corrected line
decomposition.plot()
plt.show()

plot_acf(data['AMZN'], lags=30)  # Changed 'data' to 'data['AMZN']'
plt.show()

train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

model = ARIMA(train_data['AMZN'], order=(5,1,0))
fit_model = model.fit()
forecast = fit_model.forecast(steps=len(test_data))[0]

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['AMZN'], label='Train')
plt.plot(test_data.index, test_data['AMZN'], label='Test')
plt.plot(test_data.index, forecast, label='Forecast')
plt.title('ARIMA Forecasting for AMZN Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
