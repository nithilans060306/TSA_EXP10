# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 17-11-25

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('gold.csv')
data['Price'] = data['Price'].astype(str).str.replace(',', '').astype(float)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

plt.plot(data['Date'], data['Price'])
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('Gold Price Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Price'])

plot_acf(data['Price'])
plt.show()

plot_pacf(data['Price'])
plt.show()

sarima_model = SARIMAX(data['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['Price'][:train_size], data['Price'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('SARIMA Model Predictions for Gold Price')
plt.legend()
plt.show()
```
### OUTPUT:

#### Original Data:
<img width="719" height="569" alt="image" src="https://github.com/user-attachments/assets/0133c20c-7d4e-4414-823e-a0119d44e0af" />

#### AutoCorrelation:
<img width="708" height="540" alt="image" src="https://github.com/user-attachments/assets/15770eb1-5024-4109-9ebf-a1fb9abbfa17" />

#### PartialCorrelation:
<img width="707" height="540" alt="image" src="https://github.com/user-attachments/assets/6d9cdddb-0210-41bc-bc92-af1874f355bd" />

#### Model Prediction:
<img width="727" height="563" alt="image" src="https://github.com/user-attachments/assets/9db999c7-d49e-4400-8fb4-70d9cf31c2bd" />



### RESULT:
Thus the program run successfully based on the SARIMA model.
