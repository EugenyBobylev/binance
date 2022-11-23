import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf


def draw_first_plot():
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


def draw_secont_plot(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (15, 5)
    df["Adj Close"].plot(title="Apple's stock in 2020")
    plt.show()


def draw_bolinger_plot(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (15, 5)
    df = ta.bbands(df["Adj Close"], length=20, talib=False)

    df_blg = df[["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]]
    df_blg.plot(title="Bolinger Bands (pandas_ta).")
    plt.show()


def pta_print_indicators():
    df = pd.DataFrame()
    print(df.ta.indicators())


if __name__ == '__main__':
    # draw_first_plot()
    _df: pd.DataFrame = yf.download("AAPL", start="2020-01-01", end="2021-12-31")
    draw_secont_plot(_df)
    draw_bolinger_plot(_df)
    # pta_print_indicators()
