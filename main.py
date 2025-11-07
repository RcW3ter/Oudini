from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error ,mean_squared_error
import yfinance as yf
import os
import pandas as pd
import datetime

type_ = 'd'
ticker = 'BTC-EUR'

if os.path.exists(f'cache\\{ticker}.parquet'):
    data = pd.read_parquet(f"cache\\{ticker}.parquet")
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    if today in str(data.index.max()) :
        print('State : Cache valide')

    else :
        t = data.index.max() - datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        print(f'State : {t} manquent , en cours de téléchargement')

        data_ = yf.download(ticker, start=data.index.max(), interval=f'1{type_}', auto_adjust=True)
        data = pd.concat([data, data_])

        data.to_parquet(f"cache\\{ticker}.parquet", index=True)
        print('State : Cache mis a jours')

else : 
    data = yf.download(ticker, period="max", interval=f'1{type_}', auto_adjust=True)
    data.to_parquet(f"cache\\{ticker}.parquet", index=True)

X = data[['Open', 'High', 'Low', 'Volume']].copy()
X['return_1h'] = data['Close'].pct_change()
X['ma_7'] = data['Close'].rolling(7).mean()
X['volatility'] = data['Close'].pct_change().rolling(7).std()

Y = data['Close'].shift(-1)

df = X.copy()
df['Y'] = Y
df = df.dropna()

X_values = df.loc[:, df.columns != 'Y'].values
y_values = df['Y'].values.ravel()

tscv_gap = TimeSeriesSplit(n_splits=9, test_size=120, gap=3)
RFR = RandomForestRegressor(n_estimators=350)

print("\n=== TimeSeriesSplit ===")
for i, (train_index, test_index) in enumerate(tscv_gap.split(X_values)):
    print(f"\nFold {i+1}")
    print(f"Train indices: {train_index[0]} -> {train_index[-1]} | Test indices: {test_index[0]} -> {test_index[-1]}")

    X_train, X_test = X_values[train_index], X_values[test_index]
    y_train, y_test = y_values[train_index], y_values[test_index]

    RFR.fit(X_train, y_train)
    y_pred = RFR.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

RFR.fit(X_values, y_values)

print("\nScore R² :", RFR.score(X_values, y_values))

x_new = X_values[-1].reshape(1, -1)  
y_pred = RFR.predict(x_new)
y_pred = float(y_pred[0])

print(f"Prediction : {y_pred:.2f} €")
