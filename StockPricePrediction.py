import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Set display options
pd.set_option('display.max_columns', None)

# Get S&P 500 data
sp500 = yf.Ticker("^GSPC").history(period="max")

# Remove unwanted columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create target variable and shift it to get predictions for next day
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Choose data from 1990 onwards
sp500 = sp500.loc["1990-01-01":].copy()

# Define the model
model = RandomForestClassifier(n_estimators=300, min_samples_split=2, random_state=42)

# Split data into training and test sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Fit the model
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

# Make predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Calculate precision score
precision = precision_score(test["Target"], preds)
print("Precision Score:", precision)

# Plot actual vs predicted values
combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ["Target", "Predictions"]
combined.plot()

# Function to make predictions
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Function to backtest
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Backtest predictions
predictions = backtest(sp500, model, predictors)
print(predictions["Predictions"].value_counts())

# Calculate precision score for backtested predictions
precision_backtest = precision_score(predictions["Target"], predictions["Predictions"])
print("Precision Score (Backtest):", precision_backtest)

# Print target distribution
print("Target Distribution:", predictions["Target"].value_counts() / predictions.shape[0])

# Define horizons for rolling calculations
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

# Create new predictors based on rolling averages and trends
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

# Drop rows with NaN values
sp500 = sp500.dropna()

# Print data
print(sp500)

# Plot S&P 500 Index
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(sp500.index, sp500["Close"], label="Close Price")
plt.title('S&P 500 Index', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price USD ($)', fontsize=16)
plt.show()
