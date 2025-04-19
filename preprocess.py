import yfinance as yf
import ta
import pandas as pd

def preprocess(data):
    data = data.dropna()

    # lag features
    data['lag_1'] = data['Close'].shift(1)
    data['lag_3'] = data['Close'].shift(3)
    data['lag_7'] = data['Close'].shift(7)

    data['pct_change'] = data['Close'].pct_change()

    data['rolling_mean_week'] = data['Close'].rolling(7).mean()
    data['rolling_std_week'] = data['Close'].rolling(7).std()

    data['range'] = data['High'] - data['Low']

    # financial indicators
    close = data['Close']
    data['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    data['macd'] = ta.trend.MACD(close).macd()
    data['ema_10'] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    data['bb_bbm'] = ta.volatility.BollingerBands(close).bollinger_mavg()

    # date information
    date = data['Date']
    data = data.drop(columns=['Date'])
    date = pd.to_datetime(date, yearfirst=True)
    data['MONTH'] = date.dt.month 
    data['DAY_OF_WEEK'] = date.dt.day_of_week
    data['YEAR'] = date.dt.year
    data['QUARTER'] = date.dt.quarter
    data['DAY_OF_MONTH'] = date.dt.day
    data['DAY_OF_YEAR'] = date.dt.dayofyear
    data['IS_YEAR_START'] = date.dt.is_year_start
    data['IS_YEAR_END'] = date.dt.is_year_end
    data['IS_QUARTER_START'] = date.dt.is_quarter_start
    data['IS_QUARTER_END'] = date.dt.is_quarter_end

    return data

def main():
    columns = ['Date','Close','High','Low','Open','Volume']
    btc = pd.read_csv('btc.csv', skiprows=3, names=columns)

    btc = preprocess(btc)

    btc.to_csv('new_btc.csv')

if __name__ == "__main__":
    main()