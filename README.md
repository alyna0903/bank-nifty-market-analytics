**Bank Nifty Market Analytics & Forecasting**

A Python-based data analytics project exploring the historical performance, trends, risk, volatility, and short-term price behavior of the Bank Nifty index.

The project combines exploratory data analysis, financial performance analysis, data visualization, and a lightweight machine learning component for next-day closing price forecasting.

**Project Overview**

This project analyzes historical Bank Nifty market data from January 2006 to December 2019.

The primary focus is on understanding long-term market behavior and identifying patterns in price movements, returns, volatility, drawdowns, and risk-adjusted performance.

A small machine learning component is also included to compare different Support Vector Regression (SVR) models for next-day closing price forecasting.

**Dataset**

Period: January 2006 – December 2019

Records: 3,448 trading days

Features: 6

The dataset contains the following variables:

| **Column** | **Description**                      |
| ---------- | ------------------------------------ |
| datetime   | Trading date                         |
| open       | Opening price                        |
| high       | Highest price during the trading day |
| low        | Lowest price during the trading day  |
| close      | Closing price                        |
| volume     | Trading volume                       |

Note: The dataset contains zero values for volume, so volume was not used as an analytical feature in this project.

**Key Analysis**

The project includes:

• Dataset exploration and data cleaning  
• Descriptive statistics  
• Closing price analysis  
• Daily return analysis  
• 20, 50, 100 and 200-day moving averages  
• 30-day rolling volatility  
• Drawdown analysis  
• Monthly performance analysis  
• Yearly performance analysis  
• OHLC correlation analysis  
• Risk and return metrics  
• Best and worst trading day analysis

**Key Findings**

**Market Performance**

The analysis shows a strong long-term upward movement in Bank Nifty over the period studied.

• Cumulative price return: 608.49%  
• Annualized price return: 15.04%  
• Average closing price: 14,007.34  
• Median closing price: 11,362.03  
• Maximum closing price: 32,384.95  
• Minimum closing price: 3,339.70

The yearly average closing price increased substantially over the period, although the index experienced temporary declines during periods of significant market stress.

**Risk and Volatility**

The analysis also highlights the substantial risk associated with the index.

• Annualized volatility: 29.34%  
• Maximum drawdown: −68.78%  
• Best trading day: +18.81%  
• Worst trading day: −14.75%  
• Positive trading days: 52.71%

The large maximum drawdown demonstrates that strong long-term returns were accompanied by periods of significant capital decline.

**Risk-Adjusted Performance**

| **Metric**            | **Value** |
| --------------------- | --------- |
| Annualized Return     | 15.04%    |
| Annualized Volatility | 29.34%    |
| Sharpe Ratio          | 0.513     |
| Sortino Ratio         | 0.753     |
| Maximum Drawdown      | −68.78%   |

The Sortino Ratio is higher than the Sharpe Ratio, indicating that the return profile appears relatively better when evaluated specifically against downside volatility.

**Machine Learning Forecasting**

A lightweight machine learning component was added to evaluate whether Support Vector Regression could improve next-day closing price prediction.

The following models were compared:

• Linear SVR  
• Polynomial SVR  
• RBF SVR  
• Naive Baseline

The models were evaluated using:

• Mean Absolute Error (MAE)  
• Root Mean Squared Error (RMSE)  
• R² Score

**Model Comparison**

| **Model**      | **MAE**   | **RMSE**  | **R²**   |
| -------------- | --------- | --------- | -------- |
| Linear SVR     | 204.60    | 283.84    | 0.9857   |
| Naive Baseline | 204.98    | 284.00    | 0.9856   |
| RBF SVR        | 9,791.11  | 11,028.14 | −20.6485 |
| Polynomial SVR | 14,142.88 | 15,806.08 | −43.4705 |

**Forecasting Insight**

The Linear SVR achieved the best performance among the tested SVR models and performed only marginally better than the naive baseline.

The RBF and Polynomial SVR models performed substantially worse.

This suggests that the nonlinear SVR models did not generalize effectively with the current feature set and methodology.

The comparison with the naive baseline is particularly important because it shows that a high R² value alone does not necessarily indicate that a forecasting model provides meaningful predictive improvement.

The machine learning component is therefore intended as an exploratory forecasting exercise rather than a production trading model.

**Visualizations**

The project generates the following visualizations:

1. Bank Nifty Closing Price and Moving Averages
2. Daily Returns
3. 30-Day Rolling Volatility
4. Drawdown Analysis
5. Monthly Average Closing Price
6. Yearly Average Closing Price
7. OHLC Correlation Matrix
8. Actual vs Predicted Closing Prices

The generated visualizations are available in the images folder.

**Project Structure**

Bank-Nifty-Market-Analytics-Exploratory-Forecasting/

│  
├── data/  
│ ├── banknifty.csv  
│ └── README.md  
│  
├── images/  
│ ├── 01_price_moving_averages.png  
│ ├── 02_daily_returns.png  
│ ├── 03_rolling_volatility.png  
│ ├── 04_drawdown.png  
│ ├── 05_monthly_average_close.png  
│ ├── 06_yearly_average_close.png  
│ ├── 07_correlation_heatmap.png  
│ ├── 08_predictions_vs_actual.png  
│ └── README.md  
│  
├── report/  
│ ├── model_comparison.csv  
│ ├── monthly_performance.csv  
│ ├── predictions.csv  
│ ├── svr_predictions.csv  
│ ├── yearly_performance.csv  
│ └── README.md  
│  
├── src/  
│ ├── analytics.py  
│ ├── main.py  
│ ├── prediction.py  
│ ├── visualizations.py  
│ └── README.md  
│  
├── .gitignore  
├── requirements.txt  
└── README.md

**Technologies Used**

Python

Pandas — Data manipulation and analysis

NumPy — Numerical computation

Matplotlib — Data visualization

Seaborn — Statistical visualization

Scikit-learn — Machine learning and model evaluation

**How to Run**

**1\. Clone the Repository**

git clone <https://github.com/alyna0903/Bank-Nifty-Market-Analytics-Exploratory-Forecasting.git>

cd Bank-Nifty-Market-Analytics-Exploratory-Forecasting

**2\. Install Dependencies**

pip install -r requirements.txt

**3\. Run the Project**

python src/main.py

The script performs the complete analysis, generates the visualizations, evaluates the forecasting models, and saves the resulting CSV files in the report folder.

**Limitations**

• The dataset ends in December 2019 and therefore does not represent current market conditions.

• The dataset contains no meaningful volume information, so volume-based analysis was not performed.

• The forecasting models use a limited set of features.

• The forecasting task focuses on predicting the next-day closing price rather than directly predicting returns or trading signals.

• The Linear SVR provides only a marginal improvement over the naive baseline.

• The forecasting results should not be interpreted as evidence of reliable trading performance.

• The project is intended for analytical and educational purposes rather than investment decision-making.

**Future Improvements**

• Add technical indicators such as RSI, MACD, and Bollinger Bands

• Incorporate meaningful volume data

• Include macroeconomic and broader market variables

• Implement walk-forward validation

• Perform systematic hyperparameter tuning

• Compare additional machine learning models

• Experiment with time-series forecasting approaches

• Evaluate predictions using returns rather than only price levels

• Develop a more robust feature engineering pipeline

**Disclaimer**

This project is intended for educational and analytical purposes only and does not constitute financial or investment advice.
