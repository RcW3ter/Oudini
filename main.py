from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import yfinance as yf
import os
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

type_ = 'wk'
ticker = 'BTC-EUR'
path = f'cache/{ticker}.parquet'

if os.path.exists(path):
    data = pd.read_parquet(path)
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    if today in str(data.index.max()):
        print('State : Cache valide')
    else:
        print('State : Cache mis à jour')
        data_ = yf.download(ticker, start=data.index.max(), interval=f'1{type_}', auto_adjust=True)
        data = pd.concat([data, data_])
        data.to_parquet(path, index=True)
else:
    data = yf.download(ticker, period="max", interval=f'1{type_}', auto_adjust=True)
    data.to_parquet(path, index=True)

data["return"] = data["Close"].pct_change()
data["ma_7"] = data["Close"].rolling(7).mean()
data["volatility"] = data["Close"].pct_change().rolling(7).std()

data["target"] = data["Close"].pct_change().shift(-1)

df = data[['Open', 'High', 'Low', 'Volume', 'return', 'ma_7', 'volatility', 'target']].dropna().reset_index(drop=True)

X = df.drop('target', axis=1)
y = df['target']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=8, test_size=120, gap=3)
model = RandomForestRegressor(n_estimators=250, random_state=42)

mae_list, rmse_list = [], []

print("\n=== TimeSeriesSplit ===")
for i, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_list.append(mae)
    rmse_list.append(rmse)

    print(f"\nFold {i+1}")
    print(f"Train: {train_idx[0]}→{train_idx[-1]} | Test: {test_idx[0]}→{test_idx[-1]}")
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

plt.plot(range(1, 9), mae_list, marker='o', label='MAE')
plt.plot(range(1, 9), rmse_list, marker='o', label='RMSE')
plt.title("Erreur par fold (BTC variations)")
plt.xlabel("Fold")
plt.ylabel("Erreur")
plt.legend()
plt.show()

model.fit(X_scaled, y)
score = model.score(X_scaled, y)
print(f"\nScore R² global : {score:.4f}")

x_new = X_scaled[-1].reshape(1, -1)
pred_return = model.predict(x_new)[0]
last_close = float(data["Close"].iloc[-1].item())

pred_price = last_close * (1 + pred_return)

print(f"\nProchaine variation prévue : {pred_return*100:.3f}%")
print(f"Dernier prix : {last_close:.2f} €")
print(f"Prix prévu : {pred_price:.2f} €")

