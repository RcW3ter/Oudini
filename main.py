from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

data = yf.download('BTC-EUR', period="max", interval='1d', auto_adjust=True)

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
