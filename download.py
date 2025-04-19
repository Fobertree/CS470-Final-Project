import yfinance as yf
 

def main():
    btc = yf.download('BTC-USD', start='2020-01-01', end='2025-04-16')
    btc.to_csv('btc.csv')

if __name__ == "__main__":
    main()