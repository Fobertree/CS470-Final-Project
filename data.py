import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

def get_tkr_data(tkr, period='6mo', norm=False, csv_path=None):
    df = yf.download(tkr, period=period)

    if csv_path:
        # os.makedirs(os.path.dirname(csv_path), exist_ok=True)  
        df.to_csv(csv_path)

    if norm:
        scaler = MinMaxScaler()
        df.values = scaler.fit_transform(df.values.reshape(-1, 1)).flatten()
            
    return df

if __name__ == "__main__":
    get_tkr_data("BTC-USD", csv_path="Data/data.csv")
