import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Download stock data
ticker = 'AAPL'  # Apple Inc.
df = yf.download(ticker, start="2018-01-01", end="2023-01-01")

# Step 2: Preprocessing
df = df[['Close']].copy()
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Step 3: Train-test split
X = df[['Close']].values
y = df['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Step 4: Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict
predicted = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predicted))
print(f"RMSE: {rmse:.2f}")

# Step 6: Plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(predicted)), predicted, label='Predicted')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
