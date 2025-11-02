from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import os
import pandas as pd
import datetime

type_ = 'd'

if os.path.exists('cache\\BTC-EUR.parquet'):
    data = pd.read_parquet("cache\\BTC-EUR.parquet")
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    if today in str(data.index.max()) :
        print('State : Cache valide')

    else :
        t = data.index.max() - datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        print(f'State : {t} manquent , en cours de téléchargement')

        data_ = yf.download('BTC-EUR', start=data.index.max(), interval=f'1{type_}', auto_adjust=True)
        data = pd.concat([data, data_])

        data.to_parquet("cache\\BTC-EUR.parquet", index=True)
        print('State : Cache mis a jours')

else : 
    data = yf.download('BTC-EUR', period="max", interval=f'1{type_}', auto_adjust=True)
    data.to_parquet("cache\\BTC-EUR.parquet", index=True)

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

RFR = RandomForestRegressor(n_estimators=250, random_state=42)
RFR.fit(X_values, y_values)

print("Score R² :", RFR.score(X_values, y_values))

x_new = X_values[-1].reshape(1, -1)  
y_pred = RFR.predict(x_new)
y_pred = float(y_pred[0])

print(f"Prediction : {y_pred:.2f} €")
