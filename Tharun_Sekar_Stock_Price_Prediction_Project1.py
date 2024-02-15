import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

df = pd.read_csv("SMSN.IL.csv")
df.head()

df.describe()

stock_data =df[["Date","Close"]] 


stock_data.info()

stock_data["Date"] = pd.to_datetime(stock_data["Date"])

stock_data.info()

stock_data.head()

stock_data=stock_data.set_index("Date")

stock_data.describe()

stock_data.isnull().sum()

plt.title("Distribution of stock_data")
plt.hist(stock_data.Close)

plt.style.use("ggplot")
plt.figure(figsize=(18,8))
plt.grid(True)
plt.xlabel('Dates',fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Close Prices', fontsize=20)
plt.yticks(fontsize=15)
plt.plot(stock_data["Close"],linewidth=3,color='blue')
plt.title('Samsung Stock Closing Price', fontsize=30)
plt.show()


plt.style.use("ggplot")
plt.figure(figsize=(18,8))
plt.grid(True)
plt.xlabel('Dates',fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Close Prices', fontsize=20)
plt.yticks(fontsize=15)
plt.hist(stock_data["Close"],linewidth=3,color='blue')
plt.title('Samsung Stock Closing Price', fontsize=30)
plt.show()

df_close = stock_data['Close']
df_close.plot(kind='kde',figsize=(20,10),linewidth=3)
plt.xticks(fontsize=15)
plt.grid("both")
plt.ylabel("Density",fontsize=20)
plt.yticks(fontsize=15)
plt.show()

rolmean=stock_data["Close"].rolling(48).mean()
rolstd = stock_data["Close"].rolling(48).std()

plt.plot(stock_data.Close)
plt.plot(rolmean)
plt.plot(rolstd)
plt.legend(['Stock Data', 'Rolling Mean', 'Rolling Standard Deviation'])
plt.show()

from statsmodels.tsa.stattools import adfuller
adft = adfuller(stock_data["Close"])

pd.Series(adft[0:4],index=["test stats","p-value","lag","data points"])


#Test for staionarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(48).mean() # rolling mean
    rolstd = timeseries.rolling(48).std() # rolling standard deviation
    # Plot rolling statistics:
    plt.figure(figsize = (18,8))
    plt.grid('both')
    plt.plot(timeseries, color='blue',label='Original', linewidth = 3)
    plt.plot(rolmean, color='red', label='Rolling Mean',linewidth = 3)
    plt.plot(rolstd, color='black', label = 'Rolling Std',linewidth = 4)
    plt.legend(loc='best', fontsize = 20, shadow=True,facecolor='lightpink',edgecolor = 'k')
    plt.title('Rolling Mean and Standard Deviation', fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    

test_stationarity(stock_data.Close)

from  statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(stock_data[["Close"]],period=12)
result.seasonal


fig = plt.figure(figsize=(20,10))
fig = result.plot()
fig.set_size_inches(17,10)


from  statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(stock_data[["Close"]],period=12,model="multiplicative")

fig = plt.figure(figsize=(20,10))
fig = result.plot()
fig.set_size_inches(17,10)


max_close_value=np.max(stock_data['Close'])
max_close_rows = stock_data[stock_data['Close'] == max_close_value]
max_close_rows

stock_data.describe()




import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'stock_data' is your DataFrame with 'Close' column
plt.figure(figsize=(12, 4))

# ACF plot
plt.subplot(1, 2, 1)
plot_acf(stock_data['Close'], lags=30, ax=plt.gca(), title='Autocorrelation Function (ACF)')

# PACF plot
plt.subplot(1, 2, 2)
plot_pacf(stock_data['Close'], lags=30, ax=plt.gca(), title='Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


df_close=stock_data["Close"]
df_close

df_close=df_close.diff()
df_close=df_close.dropna()

test_stationarity(df_close)



import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'stock_data' is your DataFrame with 'Close' column
plt.figure(figsize=(12, 4))

# ACF plot
plt.subplot(1, 2, 1)
plot_acf(df_close, lags=30, ax=plt.gca(), title='Autocorrelation Function (ACF)')

# PACF plot
plt.subplot(1, 2, 2)
plot_pacf(df_close, lags=30, ax=plt.gca(), title='Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


stock_data_reset = stock_data.reset_index()
stock_data_reset


import matplotlib.pyplot as plt

train_data = stock_data_reset.iloc[:-60]
test_data = stock_data_reset.iloc[-60:]

plt.figure(figsize=(18, 8))
plt.grid(True)
plt.xlabel('Dates', fontsize=20)
plt.ylabel('Closing Prices', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plotting using the 'Date' column as the x-axis
plt.plot(train_data['Date'], train_data['Close'], 'red', label='Train data', linewidth=5)
plt.plot(test_data['Date'], test_data['Close'], 'green', label='Test data', linewidth=5)

plt.legend(fontsize=20, shadow=True, facecolor='lightpink', edgecolor='k')
plt.show()



import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

history = [x for x in train_data.Close]
history

model=ARIMA(history,order=(1,1,1))


model = model.fit()


model.summary()


model.forecast()



train_data=train_data.set_index('Date')
train_data



test_data=test_data.set_index('Date')
train_data


train_data.index = pd.to_datetime(train_data.index)
test_data.index = pd.to_datetime(test_data.index)


test_val =test_data.Close
test_val[0]


mean_squared_error([test_val[0]],model.forecast())

np.sqrt(mean_squared_error([test_val[0]],model.forecast()))

def train_arima_model(X, y, arima_order):
    # prepare training dataset
    # make predictions list
    history = [x for x in X]
    predictions = list()
    for t in range(len(y)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(y[t])
    # calculate out of sample error
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return rmse


def evaluate_model(dataset,test,p_values,d_values,q_values):
    dataset = dataset.astype('float32')
    best_score,best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = train_arima_model(dataset,test,order)
                    if rmse < best_score:
                        best_score,best_cfg = rmse,order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg,best_score))


print(train_data.dtypes)
print(test_data.dtypes)

import warnings
warnings.filterwarnings('ignore')
p_values = range(0,3)
d_values = range(0,3)
q_values = range(0,3)
evaluate_model(train_data['Close'],test_data['Close'],p_values,d_values,q_values)



history = [x for x in train_data['Close']]
predictions = list()
for i in range(len(test_data['Close'])):
    model=ARIMA(history,order=(2,0,2))
    model = model.fit()
    fc = model.forecast(alpha=0.05)
    predictions.append(fc)
    history.append(test_data['Close'][i])
print(f"RMSE is {np.sqrt(mean_squared_error(test_data['Close'],predictions))}")


plt.figure(figsize=(18,8))
plt.grid(True)
plt.plot(range(len(test_data)),test_data, label = 'True Test Close Value', linewidth = 5)
plt.plot(range(len(predictions)), predictions, label = 'Predictions on test data', linewidth = 5)
plt.xticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.legend(fontsize = 20, shadow=True,facecolor='lightpink',edgecolor = 'k')
plt.show()


fc_series = pd.Series(predictions,index=test_data.index)
fc_series

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='Training', color = 'blue')
plt.plot(test_data, label='Test', color = 'green', linewidth = 3)
plt.plot(fc_series, label='Forecast', color = 'red')
plt.title('Forecast vs Actuals on test data')
plt.legend(loc='upper left', fontsize=8)
plt.show()

from statsmodels.graphics.tsaplots import plot_predict
fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(111)
plot_predict(result=model,start=1, end=len(stock_data)+60, ax = ax1)
plt.grid("both")
plt.legend(['Forecast','Close','95% confidence interval'],fontsize = 20, shadow=True,facecolor='lightblue',edgecolor = 'k')
plt.show()


# evaluate parameters
import warnings
warnings.filterwarnings('ignore')
history = [x for x in train_data['Close']]
predictions = list()
conf_list = list()
for t in range(len(test_data['Close'])):
    model = sm.tsa.statespace.SARIMAX(history, order = (0,1,0), seasonal_order = (1,1,1,3))
    model_fit = model.fit()
    fc = model_fit.forecast()
    predictions.append(fc)
    history.append(test_data['Close'][t])
print('RMSE of SARIMA Model:', np.sqrt(mean_squared_error(test_data['Close'], predictions)))



plt.figure(figsize=(18,8))
plt.title('Forecast vs Actual', fontsize = 25)
plt.plot(range(60), predictions, label = 'Predictions', linewidth = 4)
plt.plot(range(60), test_data['Close'], label = 'Close', linewidth = 4)
plt.legend(fontsize = 25, shadow=True,facecolor='lightpink',edgecolor = 'k')  


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from copy import deepcopy as dc
import numpy as np
import pandas as pd


# Load your stock price data
# Assuming you have 'Date' and 'Close' columns in your dataset
# Replace 'your_data.csv' with the actual file path or use your preferred way of loading data
stock_data = pd.read_csv('SMSN.IL.csv')

# Convert 'Date' to datetime and set it as index
stock_data = stock_data[['Date','Close']]
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)


# Function to prepare the dataset for LSTM
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
        
    df.dropna(inplace=True)
    
    return df



# Hyperparameters
lookback = 7
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Prepare dataset
shifted_df = prepare_dataframe_for_lstm(stock_data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

# Scaling the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X = dc(np.flip(X, axis=1))

split_index = int(len(X) * 0.95)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()




# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(1, 4, 1).to(device)




# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Training the model
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model
# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy().flatten()

# Inverse transform the predictions and actual values
y_pred = y_pred.reshape(-1, 1).flatten()
y_true = y_test.numpy().reshape(-1, 1).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f'RMSE: {rmse}')



plt.plot(y_true, label='Actual Close')
plt.plot(y_pred, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
