import io
import datetime
import numpy
import pandas
import base64
import matplotlib.pyplot as plt
from scipy.stats import norm
from enum import Enum

from helpers.logger import logger


class ValueAtRiskMethods(str, Enum):
    historical_simulation = "historical_simulation"
    linear_model = "linear_model"
    monte_carlo = "monte_carlo"


def value_at_risk():
    logger.debug("Calculating value at risk...")


def calculate_returns(data: pandas.Series) -> pandas.Series:
    """Calculate normalized returns."""
    returns = data / data.shift(-1)
    returns = returns.dropna()
    return returns


def calculate_log_returns(data: pandas.Series) -> pandas.Series:
    """Calculate normalized log returns."""
    returns = numpy.log(data / data.shift(-1))
    returns = returns.dropna()
    return returns


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def portfolio_historical_var(
    portfolio_with_data: dict,
    symbol_data: pandas.Series,
    confidence_level: float,
    horizon_days: int,
    date_from: datetime.date,
    date_to: datetime.date,
) -> float:
    portfolio_values = pandas.Series()
    for day in daterange(date_from, date_to + datetime.timedelta(1)):
        pd_day = pandas.Timestamp(day)
        portfolio_value = 0
        include_date = True
        for symbol, symbol_data in portfolio_with_data.items():
            value = symbol_data["value"]
            returns: pandas.Series = symbol_data["returns"]
            if pd_day in returns:
                portfolio_value += value * returns[pd_day]
            else:
                include_date = False
                break
        if include_date:
            portfolio_values[pd_day] = portfolio_value

    sorted_portfolio_values = numpy.sort(portfolio_values)
    percentile = 1 - confidence_level
    percentile_sample_index = int(percentile * len(sorted_portfolio_values))
    worst_portfolio_value = sorted_portfolio_values[percentile_sample_index]
    current_portfolio_value = sum(symbol_data["value"] for symbol_data in portfolio_with_data.values())
    var = (current_portfolio_value - worst_portfolio_value) * numpy.sqrt(horizon_days)
    return var


def historical_simulation_var(
    returns: pandas.Series,
    confidence_level: float,
    portfolio_value: int | float,
    historical_days: int,
    horizon_days: int,
) -> float:
    """Calculate Value at Risk using historical simulation method."""
    first_return = max(len(returns) - historical_days, 0)
    returns_subset: pandas.Series = returns[first_return:]
    sorted_returns = numpy.sort(returns_subset)
    percentile = 1 - confidence_level
    percentile_sample_index = int(percentile * len(sorted_returns))
    worst_portfolio_value = sorted_returns[percentile_sample_index] * portfolio_value
    var = (portfolio_value - worst_portfolio_value) * numpy.sqrt(horizon_days)
    return var


def linear_model_var(
    returns: pandas.Series,
    confidence_level: float,
    portfolio_value: int | float,
    historical_days: int,
    horizon_days: int,
) -> float:
    """Calculate Value at Risk using linear model simulation."""
    first_return = max(len(returns) - historical_days, 0)
    returns_subset: pandas.Series = returns[first_return:]
    # Standard deviation is the statistical measure of market volatility
    std_dev = numpy.std(returns_subset)
    standard_score = norm.ppf(confidence_level)
    var = standard_score * std_dev * portfolio_value * numpy.sqrt(horizon_days)
    return var


def monte_carlo_var(
    returns: pandas.Series,
    confidence_level: float,
    portfolio_value: int | float,
    historical_days: int,
    horizon_days: int,
    number_of_samples: int = 5000,
) -> float:
    """Calculate Value at Risk using monte carlo simulation."""
    first_return = max(len(returns) - historical_days, 0)
    returns_subset: pandas.Series = returns[first_return:]
    std_dev = numpy.std(returns_subset)
    # loc=Mean(center), scale=Std(Spread/Width)
    norm_distribution_samples = numpy.random.normal(loc=0, scale=std_dev, size=number_of_samples)
    sorted_norm_distribution_samples = numpy.sort(norm_distribution_samples)
    percentile = 1 - confidence_level
    percentile_sample_index = int(percentile * len(sorted_norm_distribution_samples))
    one_day_var = portfolio_value * sorted_norm_distribution_samples[percentile_sample_index]
    var = abs(one_day_var * numpy.sqrt(horizon_days))
    return var


def calculate_value_at_risk(
    method: str,
    data: pandas.DataFrame,
    confidence_level: float,
    portfolio_value: int | float,
    historical_days: int,
    horizon_days: int,
) -> float:
    """Calcualte Value at Risk."""
    data = data["close"]
    returns = calculate_returns(data)

    if method == ValueAtRiskMethods.historical_simulation:
        var = historical_simulation_var(returns, confidence_level, portfolio_value, historical_days, horizon_days)
    if method == ValueAtRiskMethods.historical_simulation:
        var = linear_model_var(returns, confidence_level, portfolio_value, historical_days, horizon_days)
    if method == ValueAtRiskMethods.historical_simulation:
        var = monte_carlo_var(returns, confidence_level, portfolio_value, historical_days, horizon_days)

    return var


def plot_hurst_eponent(intervals: list, data: list) -> float:
    fig, ax = plt.subplots()
    plt.xscale("log")
    plt.yscale("log")
    ax.plot(intervals, data)
    a, b = numpy.polyfit(numpy.log(intervals), numpy.log(data), 1)
    plt.plot(intervals, [numpy.exp(y) for y in [a * numpy.log(x) + b for x in intervals]])
    ax.set_xlabel("Segment length (log)", fontsize=8, labelpad=6, fontweight="bold")
    ax.set_ylabel("Mean standard deviation (log)", fontsize=8, labelpad=6, fontweight="bold")
    ax.set_title(
        "Mean standard deviation depending on segment length",
        fontsize=9,
        pad=12,
        fontweight="bold",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    return a, plot


def calculate_hurst_exponent(data: pandas.DataFrame):
    data = data["close"]
    # 1. logarytmiczna stopy zwrotu
    log_returns = calculate_log_returns(data)
    ro = []
    # 9. powtarzamy kroki <2, 8>, każdorazowo zwiększając długość przedziału m o jeden do momentu aż n osiągnie górną granicę
    intervals = range(5, int(len(log_returns - 1) / 2))
    for n in intervals:
        # 2. Dzielimy szereg stóp procentowych na m części złożonych z n elementów
        m = int(len(log_returns) / n)
        z = numpy.zeros(shape=(m, n))
        u = numpy.zeros(shape=(m, n))
        r = numpy.zeros(shape=m)
        s = numpy.zeros(shape=m)
        ro_iter = numpy.zeros(shape=m)
        for i in range(m):
            y_mean = numpy.mean(log_returns[i * n : (i + 1) * n])
            for j in range(n):
                # 3.Definiujemy z_ij
                z[i][j] = log_returns[i * n + j] - y_mean
                # 4.Ciag sum czesciowych
                u[i][j] = numpy.sum(z[i])
            # 5. Liczymy odchylenie standardowe
            s[i] = numpy.std(z[i])
            # 6. Określamy zakres i-tego przedziału
            r[i] = numpy.max(u[i] - numpy.min(u[i]))
        # 7. Normalizujemy wartości i-tego przedziału
        ro_iter = r / s
        # 8. Obliczamy średnią wartość znormalizowanego i-tego przedziału
        ro.append(numpy.mean(ro_iter))
    # 10,11. nachylenie prostej średniego odchylenia standardowego zależnego od długości segmentów na skali logarytmicznej to wykladnik Hursta
    hurst_exponent, hurst_plot = plot_hurst_eponent(intervals, ro)
    print(hurst_exponent)
    return hurst_exponent, hurst_plot
