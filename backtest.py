from backtesting import Backtest, Strategy
from backtesting.test import BTCUSD
import numpy as np


def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='X').values


def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(48).shift(-48)  # Returns after roughly two days
    y[y.between(-.004, .004)] = 0             # Devalue returns smaller than 0.4%
    y[y > 0] = 1
    y[y < 0] = -1
    return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

# https://kernc.github.io/backtesting.py/doc/examples/Trading%20with%20Machine%20Learning.html

class BuyLowSellHigh(Strategy):
    def __init__(self, N_TRAIN = 300):
        self.clf = None # TODO: replace with actual classifier
        


if __name__ == "__main__":
    # backtest
    # bt = backtest()
    # bt.run
    pass