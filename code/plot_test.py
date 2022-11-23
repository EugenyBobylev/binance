import pandas as pd
import yfinance as yf
# import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = yf.download("AAPL", start="2020-01-01", end="2020-12-31")
    print(len(df))
