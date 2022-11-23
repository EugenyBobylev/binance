import yfinance as yf
import pandas as pd


def print_info(ticker: yf.Ticker):
    info: dict = ticker.info

    for (key, val) in info.items():
        if val is not None:
            print(f'{key}: {val}')


def print_history(ticker: yf.Ticker):
    df: pd.DataFrame = msft.history(period='1mo', interval="60m", start='2022-01-01', end='2022-02-01')
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
        'display.width', 1000
    ):
        print(df)


if __name__ == '__main__':
    symbol = 'MSFT'
    msft: yf.Ticker = yf.Ticker(symbol)
    # print_info(msft)
    print_history(msft)
