import numpy as np
import pandas as pd
import yfinance as yf
# import pandas_datareader.data as web
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import date

plt.style.use('fivethirtyeight')
yf.pdr_override()


def get_portfolio(symbol: str, start, end=date.today()):
    _data = yf.download(symbol, data_source='yahoo', start=start, end=end)
    return _data


class StrategySma:

    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start_date = start
        self.end_date = end
        self.data = get_portfolio(self.symbol, self.start_date, self.end_date)
        self.data['SMA 30'] = ta.sma(self.data['Close'], 30)
        self.data['SMA 100'] = ta.sma(self.data['Close'], 100)
        self.data['Buy_Signal_price'], self.data['Sell_Signal_price'] = self.buy_sell()

    def buy_sell(self):
        signal_buy = []
        signal_sell = []
        position = False

        for i in range(len(self.data)):
            signal_buy.append(np.nan)
            signal_sell.append(np.nan)
            if not position and self.data['SMA 30'][i] > self.data['SMA 100'][i]:
                signal_buy[-1] = self.data['Adj Close'][i]
                position = True
                continue
            if position and self.data['SMA 30'][i] < self.data['SMA 100'][i]:
                signal_sell[-1] = self.data['Adj Close'][i]
                position = False
        return pd.Series([signal_buy, signal_sell])

    def visualization(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.data['Adj Close'], label=self.symbol, linewidth=1, color='blue', alpha=0.9, )
        ax.plot(self.data['SMA 30'], label='SMA30', linewidth=1.2, alpha=0.7)
        ax.plot(self.data['SMA 100'], label='SMA100', linewidth=1.2, alpha=0.7)
        ax.scatter(self.data.index, self.data['Buy_Signal_price'], label='Buy', marker='^', color='green', alpha=1)
        ax.scatter(self.data.index, self.data['Sell_Signal_price'], label='Sell', marker='v', color='red', alpha=1)
        ax.set_title(self.symbol + " Price History with buy and sell signals", fontsize=16)
        ax.set_xlabel(f'{self.start_date} - {self.end_date}', fontsize=14)
        ax.set_ylabel('Close Price INR (₨)', fontsize=14)
        legend = ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    sma = StrategySma('TATAMOTORS.NS', '2017-01-01', date.today())
    print(sma.data)
    sma.visualization()
