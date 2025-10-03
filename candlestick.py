from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os
from dateutil.relativedelta import relativedelta

##########__Donnee__##########

ticker = 'BTC-EUR'
data = yf.download(ticker, period="max",interval="1h",auto_adjust=True)

data.to_csv(f'csv\\{ticker}.csv',index=True)
print(data.tail(),'\n')

##########__Traitement__##########

X = data[['Open', 'High', 'Low', 'Volume']] 
Y = data['Close'].shift(-1)

X = X[:-1]
Y = Y[:-1]

##########__Trainning__##########

X_values = X.values
y_values = Y.values      

model = LinearRegression()
model.fit(X_values,y_values)

y_pred = model.predict(X_values)

print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
print("Prédictions sur X :", y_pred,'\n')

##########__Metric__##########

r2 = r2_score(Y, y_pred)
print(f"R² : {r2}")

mse = mean_squared_error(Y, y_pred)
print(f"MSE : {mse}")

rmse = np.sqrt(mse)
print(f"RMSE : {rmse}\n")

##########__Demain__##########

latest_data = X.iloc[-1].values.reshape(1, -1) 
predicted_close_today = model.predict(latest_data)

print("Prédiction du prix de clôture :", predicted_close_today[0])

##########__Prevision__##########

close_value = predicted_close_today.item()

List = pd.DataFrame({
    'Date': [datetime.datetime.today()],
    'Close': [close_value],
    'RMSE': [rmse]})

if not os.path.exists(f'prevision\\Oudini-{ticker}.csv'):
    List.to_csv(f'prevision\\Oudini-{ticker}.csv', index=False)
else:
    List.to_csv(f'prevision\\Oudini-{ticker}.csv', index=False, mode='a', header=False)