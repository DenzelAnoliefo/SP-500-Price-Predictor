# S&P 500 Stock Price Prediction with Machine Learning

  ## Introductionnnn

  This project aims to predict the direction of the S&P 500 stock index using machine learning techniques. The S&P 500 is a widely followed index that tracks the performance of 500 large-cap American stocks. By predicting its movement, this    project offers insights valuable for investors and traders.

  ## Project Overview

  This code provides a robust framework for predicting the direction of the S&P 500 stock index using machine learning, specifically a Random Forest Classifier. It begins by using the yfinance library to fetch historical S&P 500 data, focusing on key financial metrics such as Open, High, Low, Close, and Volume, while removing less relevant columns like Dividends and Stock Splits. The target variable is created to indicate whether the next day's closing price will be higher than the current day's, thus framing the problem as a binary classification task. The dataset is filtered to include data from 1990 onwards, ensuring relevance to contemporary market conditions, and is split into training and test sets, with the most recent 100 data points reserved for testing. The model is trained using a set of predictors and evaluated using precision score, achieving a precision of 60% on the test data. The code also includes functions for making predictions and backtesting, where the backtested predictions yield a precision score of 57%, demonstrating the model's robustness and reliability for real-world usage. Additional features based on rolling averages and trends are created to enhance predictive accuracy. Throughout, the code ensures no data leakage, meaning the precision scores are valid indicators of real-world performance, making it a powerful tool for trading and quantitative research.

  ## Dependencies

  Python 3
  yfinance for retrieving historical S&P 500 data.
  pandas for data manipulation and analysis.
  matplotlib for data visualization.
  scikit-learn for implementing machine learning models.

  ## Results

  The model achieves a precision score of 0.6 (60%) on the test data with no leakage.
  Backtesting the model on historical data yields a precision score of 0.57 (57%) with no leakage. Performs great for real-world usage.

  ## Contact

  Author:

  #### Denzel Anoliefo
