import math

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


class StrategyMacd:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start_date = start
        self.end_date = end
        self.data = get_portfolio(self.symbol, self.start_date, self.end_date)

    def use_for_buy(self):
        macd_buy = ta.macd(self.data['Close'])
        self.data = pd.concat([self.data, macd_buy], axis=1)

    def use_for_sell(self):
        macd_sell = ta.macd(self.data['Close'], fast=8, slow=17)
        self.data = pd.concat([self.data, macd_sell], axis=1)

    def calculate(self, risk: float = 0.025):
        self.data['MACD_Buy_Signal_price'], self.data['MACD_Sell_Signal_price'] = self.buy_sell(risk=risk)

    def buy_sell(self, risk):
        signal_buy = []
        signal_sell = []
        position = False

        for i in range(0, len(self.data)):
            signal_buy.append(np.nan)
            signal_sell.append(np.nan)
            if not position and self.data['MACD_12_26_9'][i] > self.data['MACDs_12_26_9'][i]:
                signal_buy[-1] = self.data['Adj Close'][i]
                position = True
            elif position and self.data['MACD_12_26_9'][i] < self.data['MACDs_12_26_9'][i]:
                signal_sell[-1] = self.data['Adj Close'][i]
                position = False
            elif position and self.data['Adj Close'][i] < signal_buy[-1] * (1 - risk):
                signal_sell[-1] = self.data["Adj Close"][i]
                position = False
            elif position and self.data['Adj Close'][i] < self.data['Adj Close'][i - 1] * (1 - risk):
                signal_sell[-1] = self.data["Adj Close"][i]
                position = False

        return pd.Series([signal_buy, signal_sell])

    def __color(self):
        macd_color = []
        for i in range(0, len(self.data)):
            if self.data['MACDh_12_26_9'][i] > self.data['MACDh_12_26_9'][i - 1]:
                macd_color.append(True)
            else:
                macd_color.append(False)
        return macd_color

    def visualization(self):
        self.data['positive'] = self.__color()
        plt.rcParams.update({'font.size': 10})
        fig, ax1 = plt.subplots(figsize=(18, 8))
        fig.suptitle(self.symbol, fontsize=10)
        ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)

        ax2 = plt.subplot2grid((14, 12), (10, 0), rowspan=6, colspan=14)
        ax1.set_ylabel('Price in ₨')
        ax1.plot('Adj Close', data=self.data, label='Close Price', linewidth=0.5, color='blue')
        ax1.scatter(self.data.index, self.data['MACD_Buy_Signal_price'], color='green', marker='^', alpha=1)
        ax1.scatter(self.data.index, self.data['MACD_Sell_Signal_price'], color='red', marker='v', alpha=1)
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel('Date', fontsize=8)

        ax2.set_ylabel('MACD', fontsize=8)
        ax2.plot('MACD_12_26_9', data=self.data, label='MACD', linewidth=0.5, color='blue')
        ax2.plot('MACDs_12_26_9', data=self.data, label='signal', linewidth=0.5, color='red')
        ax2.bar(self.data.index, 'MACDh_12_26_9', data=self.data, label='Volume',
                color=self.data.positive.map({True: 'g', False: 'r'}), width=1, alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax2.grid()
        plt.show()


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


def test_strategy_sma(symbol):
    strat = StrategySma(symbol, '2017-01-01', date.today())
    print(strat.data)
    strat.visualization()


def test_strategy_macd(symbol):
    strat = StrategyMacd(symbol, '2017-01-01', date.today())
    strat.use_for_buy()
    strat.calculate(0.5)
    strat.visualization()


if __name__ == '__main__':
    _symbol = 'TATAMOTORS.NS'
    test_strategy_macd(_symbol)
