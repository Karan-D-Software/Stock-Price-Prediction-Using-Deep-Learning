# 📈 Stock Price Prediction Using Deep Learning


## Gathering Data and Determining Method of Data Collection and Provenance

### Data Source
For this project, we are using historical stock price data for SPY (SPDR S&P 500 ETF) sourced from [Yahoo Finance](https://ca.finance.yahoo.com/quote/SPY/history). This dataset includes the daily Open, High, Low, Close, and Volume data for SPY, which will be used to analyze and predict future stock prices.

### Method of Data Collection
The data was collected by downloading the historical stock price CSV file from Yahoo Finance. This file contains detailed historical records of SPY's trading data, providing a rich source of information for our analysis and model training.

### Provenance of the Data
The data is provided by Yahoo Finance, a reputable source known for its comprehensive and accurate financial data. Yahoo Finance compiles and disseminates this information, ensuring it is up-to-date and reliable for financial analysis.

### Problem Description
The primary objective of this project is to predict the future closing prices of SPY using deep learning techniques. By analyzing historical price trends and patterns, we aim to build a model that can provide accurate price forecasts, which can be valuable for investors and traders.

## Identifying a Deep Learning Problem

### Problem Description
The primary deep learning problem for this project is time series forecasting, specifically predicting the future closing prices of the SPY ETF based on historical price data. Time series forecasting is crucial in finance as it helps investors and traders make informed decisions by predicting future price movements.

### Deep Learning Approach
We will employ various deep learning models to tackle this problem, focusing on model building, evaluation, and comparison. The models we plan to use include:

1. **Recurrent Neural Networks (RNNs)**:
   - **Long Short-Term Memory (LSTM) Networks**: LSTMs are well-suited for time series data as they can learn long-term dependencies and patterns in the data.
   - **Gated Recurrent Units (GRUs)**: GRUs are a simpler alternative to LSTMs and can also handle long-term dependencies effectively.

2. **Convolutional Neural Networks (CNNs)**:
   - CNNs can be applied to time series data by treating the time series as a one-dimensional image. This can capture local dependencies in the data.

3. **Autoencoders**:
   - Autoencoders can be used for anomaly detection, feature extraction, and denoising in the stock price data. By compressing and then reconstructing the data, autoencoders can help in identifying underlying patterns.

### Experiment and Comparison
We will run experiments to compare the performance of different models and algorithms. This will involve:
1. **Model Training and Evaluation**: Training each model on the historical SPY data and evaluating its performance using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
2. **Hyperparameter Tuning**: Optimizing the hyperparameters for each model to achieve the best performance.
3. **Model Comparison**: Comparing the results of different models to identify the best performing model for our forecasting task.



