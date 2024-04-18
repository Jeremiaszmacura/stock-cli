import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum


class PlotTypes(str, Enum):
    linear_plot = "linear_plot"
    candle_plot = "candle_plot"


def linear_plot(data: pandas.DataFrame, ax):
    shift = numpy.linspace(0, 6)
    for _ in shift:
        ax.plot(data["close"], color="#00ccff", linewidth=0.5)
    plt.show()


def candle_stick_plot(data: pandas.DataFrame):
    plt.subplots_adjust(bottom=0.20)
    plt.xticks(rotation=70, fontsize=6)
    plt.yticks(fontsize=8)
    up = data[data.close >= data["open"]]
    down = data[data["close"] < data["open"]]
    up_color = "#89ff00"
    up_shadow_color = "#4CAE50"
    down_color = "#ff005e"
    down_shadow_color = "#9C2525"
    bar_width = 0.5
    shadow_width = 0.2
    # Plotting up prices of the stock
    plt.bar(up.index, up.close - up.open, bar_width, bottom=up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, shadow_width, bottom=up.close, color=up_shadow_color)
    plt.bar(up.index, up.low - up.open, shadow_width, bottom=up.open, color=up_shadow_color)
    # Plotting down prices of the stock
    plt.bar(down.index, down.close - down.open, bar_width, bottom=down.open, color=down_color)
    plt.bar(down.index, down.high - down.open, shadow_width, bottom=down.open, color=down_shadow_color)
    plt.bar(down.index, down.low - down.close, shadow_width, bottom=down.close, color=down_shadow_color)


def plot_data(data: pandas.DataFrame, plot_type: PlotTypes, company_symbol: str):
    plt.style.use("dark_background")
    _, ax = plt.subplots()
    ax.set_title(company_symbol)
    plt.rc("font", size=8)
    if plot_type == PlotTypes.linear_plot:
        linear_plot(data, ax)
