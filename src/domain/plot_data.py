import pandas
import numpy
import matplotlib


def linear_plot(data: pandas.DataFrame, ax: matplotlib.axes.Axes):
    shift = numpy.linspace(0, 6)
    for _ in shift:
        ax.plot(data["close"], color="#00ccff", linewidth=0.5)
