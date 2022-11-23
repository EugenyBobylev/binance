from datetime import datetime, timedelta
from dataclasses import dataclass

from binance.spot import Spot


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


if __name__ == '__main__':
    client = Spot()
    data = client.klines('BTCUSDT', '1h')

    print(len(data))
    for candle in data:
        print(Kline.create(candle))
