from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
# from mplfinance import candlestick_ohlc

from binance.spot import Spot
from binance.cm_futures import CMFutures
from pandas.errors import DataError


@dataclass
class Kline:
    open_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    close_time: datetime
    quote_volume: float
    trade_count: int
    buy_base_asset_volume: float
    buy_quote_asset_volume: float

    def __str__(self):
        return f"open_time:'{self.open_time.strftime('%Y:%m:%d %H:%M:%S')}'; open:{self.open_price:.3f}; " \
               f"high:{self.high_price:.3f}; low:{self.low_price:.3f}; close:{self.close_price:.3f}: " \
               f"volume:{self.volume:.3f}; close_time:'{self.close_time.strftime('%Y:%m:%d %H:%M:%S')}' " \
               f"quote_volume:{self.quote_volume:.3f}, trade_count:{self.trade_count}"

    @classmethod
    def create(cls, binance_kline: list):
        return cls(
            open_time=binance_timestamp_to_utc_datetime(binance_kline[0]),
            open_price=float(binance_kline[1]),
            high_price=float(binance_kline[2]),
            low_price=float(binance_kline[3]),
            close_price=float(binance_kline[4]),
            volume=float(binance_kline[5]),
            close_time=binance_timestamp_to_utc_datetime(binance_kline[6]),
            quote_volume=float(binance_kline[7]),
            trade_count=int(binance_kline[8]),
            buy_base_asset_volume=float(binance_kline[9]),
            buy_quote_asset_volume=float(binance_kline[10])
        )


def binance_timestamp_to_utc_datetime(binance_time_stamp) -> datetime:
    # Binance timestamp is milliseconds past epoch
    epoch = datetime(1970, 1, 1, 0, 0, 0, 0)
    return epoch + timedelta(milliseconds=binance_time_stamp)


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    _df = pd.DataFrame(klines,
                       columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                'quote', 'trades', 'buy_base', 'buy_quote', 'reserved'])
    del _df['reserved']
    new_dbtypes = {
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64,
        'quote': np.float64,
        'trades': int,
        'buy_base': np.float64,
        'buy_quote': np.float64,
    }
    _df = _df.astype(new_dbtypes)
    _df['open_time'] = pd.to_datetime(_df['open_time'], unit='ms')
    _df['close_time'] = pd.to_datetime(_df['close_time'], unit='ms')
    return _df


def print_binance_data(data: list):
    print(len(data))
    for candle in data:
        print(Kline.create(candle))


def print_data_frame(data: pd.DataFrame):
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
        'display.width', 1000
    ):
        print(data)


def load_spot_data(symbol: str, interval: str = '1h') -> pd.DataFrame:
    client = Spot()
    raw_data = client.klines(symbol, interval=interval)
    _df = klines_to_dataframe(raw_data)
    return _df


def load_futures_data(symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    futures_client = CMFutures()
    raw_data = futures_client.klines(symbol, interval=interval, limit=limit)
    _df = klines_to_dataframe(raw_data)
    # print_data_frame(df)
    return _df


def ta_sma(data: pd.DataFrame, source_col_name: str, length=30) -> bool:
    col_name = f'SMA {length}'
    _ok = True
    try:
        data[col_name] = ta.sma(data[source_col_name], length)
    except KeyError:
        _ok = False
    except DataError:
        _ok = False
    return _ok


def ta_rsi(data: pd.DataFrame, source_col_name: str, look_back_period: int = 14) -> bool:
    col_name = f'RSI {look_back_period}'
    _ok = True
    try:
        data[col_name] = ta.rsi(data[source_col_name], length=look_back_period)
    except KeyError:
        _ok = False
    except DataError:
        _ok = False
    return _ok


def ta_macd(data: pd.DataFrame, source_col_name: str, fast: int, slow: int, signal: int = 9) -> (bool, pd.DataFrame):
    _ok = True
    try:
        macd = ta.macd(data[source_col_name], fast=fast, slow=slow, signal=signal)
        data = pd.concat([data, macd], axis=1)
    except KeyError:
        _ok = False, data
    except DataError:
        _ok = False, data
    return _ok, data


def ta_ao(data: pd.DataFrame, high_col_name: str, low_col_name, fast: int = 5, slow: int = 34) -> bool:
    col_name = f'AO_{fast}_{slow}'
    _ok = True
    try:
        data[col_name] = ta.ao(high=data[high_col_name], low=data[low_col_name], fast=fast, slow=slow)
    except KeyError:
        _ok = False
    except DataError:
        _ok = False
    return _ok


def vusualize_ohlc(data: pd.DataFrame):
    # get OHLC data
    ohlc: pd.DataFrame = df.loc[:, ['open_time', 'open', 'high', 'low', 'close', 'volume']]
    ohlc.rename(columns={'open_time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                         'volume': 'Volume'},   inplace=True)
    ohlc.index = pd.DatetimeIndex(ohlc['Date'])

    # CandleStick Layout, Styling
    mc = mpf.make_marketcolors(
        up='tab:blue', down='tab:red',
        edge='black',
        wick='inherit',
        # wick={'up': 'blue', 'down': 'red'},
        volume='silver',  # inherit
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridaxis='both',)

    # plot and show
    # mpf.plot(ohlc, type='candle', figsize=(18,8), volume=True, style=s)
    mpf.plot(ohlc, type='candle', figsize=(18,8), volume=True, style='default')
    mpf.show()


if __name__ == '__main__':
    start = datetime.now()
    df = load_futures_data('BTCUSD_PERP', interval='1d', limit=90)
    ta_sma(df, 'close', 30)
    ta_sma(df, 'close', 100)

    ta_rsi(df, 'close', 6)
    ta_rsi(df, 'close', 14)

    ok, df = ta_macd(df, 'close', fast=1, slow=5, signal=9)
    ta_ao(df, 'high', 'low')
    stop = datetime.now()
    print(stop - start)

    vusualize_ohlc(df)
