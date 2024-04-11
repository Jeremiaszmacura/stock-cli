import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum


class PlotTypes(str, Enum):
    linear_plot = "linear_plot"


def linear_plot(data: pandas.DataFrame, ax):
    shift = numpy.linspace(0, 6)
    for _ in shift:
        ax.plot(data["close"], color="#00ccff", linewidth=0.5)
    plt.show()


def plot_data(data: pandas.DataFrame, plot_type: PlotTypes):
    plt.style.use("dark_background")
    _, ax = plt.subplots()
    print(type(ax))
    plt.rc("font", size=8)
    if plot_type == PlotTypes.linear_plot:
        linear_plot(data, ax)
