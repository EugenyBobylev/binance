import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf


def print_data_frame(df: pd.DataFrame):
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
        'display.width', 1000
    ):
        print(df)


def draw_first_plot():
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


def draw_second_plot(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (15, 5)
    df["Adj Close"].plot(title="Apple's stock in 2020")
    plt.show()


def draw_bolinger_plot(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (15, 5)

    df = ta.bbands(df["Adj Close"], length=20, talib=False)
    df_blg = df[["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]]
    df_blg.plot(title="Bolinger Bands (pandas_ta).")

    x_data = pd.date_range('2020-01-01', periods=25, freq='MS')
    plt.grid(True)
    plt.xticks(x_data)
    plt.show()


def pta_print_indicators():
    df = pd.DataFrame()
    print(df.ta.indicators())


if __name__ == '__main__':
    # draw_first_plot()
    _df: pd.DataFrame = yf.download("AAPL", start="2020-01-01", end="2022-01-01")
    # draw_second_plot(_df)

    # x_data = pd.date_range('2020-01-01', periods=25, freq='MS')
    # print(x_data)

    # print(_df)
    # print(len(_df))
    # print(_df.index.values.tolist())
    # print(_df.loc['2020-01-02'])
    # print_data_frame(_df)
    draw_bolinger_plot(_df)
    # pta_print_indicators()
