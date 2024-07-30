"""Module used to calculate numerical statistics."""
import io
import datetime
import numpy
import pandas
import base64
from typing import Generator
import matplotlib.pyplot as plt
from scipy.stats import norm
from enum import Enum

from helpers.logger import logger


class ValueAtRiskMethods(str, Enum):
    """Enumerate data type for calculation Value at Risk methods."""

    historical_simulation = "historical_simulation"
    linear_model = "linear_model"
    monte_carlo = "monte_carlo"


class Statistics:
    def __init__(self):
        pass

    def calculate_value_at_risk(
        self,
        method: str,
        data: pandas.DataFrame,
        confidence_level: float,
        portfolio_value: int | float,
        historical_days: int,
        horizon_days: int,
    ) -> float:
        """Calculate Value at Risk (VaR) using the specified method.

        Args:
            method (str): The method to use for VaR calculation. Options are 'historical_simulation', 'linear_model', or 'monte_carlo'.
            data (pandas.DataFrame): A dataframe containing historical price data with at least a 'close' column.
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
            portfolio_value (int | float): The current value of the portfolio.
            historical_days (int): Number of historical days to consider for the simulation.
            horizon_days (int): Time horizon for VaR calculation in days.

        Returns:
            float: The calculated Value at Risk (VaR) for the portfolio over the given time horizon and confidence level.
        """
        data = data["close"]
        returns = self._calculate_returns(data)

        if method == ValueAtRiskMethods.historical_simulation:
            var = self._historical_simulation_var(
                returns, confidence_level, portfolio_value, historical_days, horizon_days
            )
        elif method == ValueAtRiskMethods.historical_simulation:
            var = self._linear_model_var(returns, confidence_level, portfolio_value, historical_days, horizon_days)
        elif method == ValueAtRiskMethods.historical_simulation:
            var = self._monte_carlo_var(returns, confidence_level, portfolio_value, historical_days, horizon_days)
        else:
            raise ValueError("Invalid method specified for VaR calculation.")

        return var

    def portfolio_historical_var(
        self,
        portfolio_with_data: dict[str, dict[str, pandas.Series]],
        symbol_data: pandas.Series,
        confidence_level: float,
        horizon_days: int,
        date_from: datetime.date,
        date_to: datetime.date,
    ) -> float:
        """Calculate Value at Risk (VaR) using historical simulation method for whole portfolio.

        Args:
            portfolio_with_data (dict): Dictionary containing portfolio data with symbols as keys and dictionaries as values.
                Each value dictionary should contain:
                    - "value": float, the current value of the position.
                    - "returns": pandas.Series, historical returns of the symbol.
            symbol_data (pandas.Series): Historical price data for the symbols in the portfolio.
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
            horizon_days (int): Time horizon for VaR calculation in days.
            date_from (datetime.date): Start date for the historical period
            date_to (datetime.date): End date for the historical period.

        Returns:
            float: The calculated Value at Risk (VaR) for the portfolio over the given time horizon and confidence level.
        """
        portfolio_values = pandas.Series(dtype=float)
        for day in self._daterange(date_from, date_to + datetime.timedelta(1)):
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

    def calculate_hurst_exponent(self, data: pandas.DataFrame):
        """
        Calculate the Hurst exponent of a time series using log returns.

        The Hurst exponent is a measure of the long-term memory of time series data.
        Values of H close to 0.5 indicate a random walk, values less than 0.5 suggest mean reversion,
        and values greater than 0.5 indicate trending behavior.

        Args:
            data (pandas.DataFrame): A dataframe containing historical price data with at least a 'close' column.

        Returns:
            float: The calculated Hurst exponent.
            matplotlib.figure.Figure: The plot of log-log scale of average rescaled range vs. interval size.

        Steps:
            1. Calculate logarithmic returns of the 'close' prices.
            2. Divide the log returns into intervals of increasing size.
            3. For each interval, compute the mean-adjusted series.
            4. Calculate the cumulative deviations from the mean (partial sums).
            5. Compute the standard deviation for each interval.
            6. Determine the range of the cumulative deviations.
            7. Normalize the range by the standard deviation.
            8. Compute the mean normalized range for each interval.
            9. Repeat for all interval sizes.
            10. Estimate the Hurst exponent as the slope of the log-log plot of the mean normalized range versus interval size.
        """
        data = data["close"]
        # 1. logarytmiczna stopy zwrotu
        log_returns = self._calculate_log_returns(data)
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
        hurst_exponent, hurst_plot = self._plot_hurst_eponent(intervals, ro)
        print(hurst_exponent)
        return hurst_exponent, hurst_plot

    def _calculate_returns(self, data: pandas.Series) -> pandas.Series:
        """Calculate normalized returns.

        Args:
            data (pandas.Series): A time series of prices or values.

        Returns:
            pandas.Series: A time series of the normalized returns.
        """
        logger.debug("Calculating time series of the normalized returns.")
        returns = data / data.shift(-1)
        returns = returns.dropna()
        return returns

    def _calculate_log_returns(self, data: pandas.Series) -> pandas.Series:
        """Calculate normalized log returns.

        Args:
            data (pandas.Series): A time series of prices or values.

        Returns:
            pandas.Series: A time series of the normalized log returns.
        """
        logger.debug("Calculating time series of the normalized log returns.")
        returns = numpy.log(data / data.shift(-1))
        returns = returns.dropna()
        return returns

    def _daterange(self, start_date: datetime.date, end_date: datetime.date) -> Generator[datetime.date, None, None]:
        """generates each date from a specified start date up to but not including a specified end date.

        Args:
            start_date (datetime.date): start date (included).
            end_date (datetime.date): end date (excluded).

        Yields:
            Generator[datetime.date, None, None]: yields datetime.date objects for each day.
        """
        for n in range(int((end_date - start_date).days)):
            yield start_date + datetime.timedelta(n)

    def _historical_simulation_var(
        self,
        returns: pandas.Series,
        confidence_level: float,
        portfolio_value: int | float,
        historical_days: int,
        horizon_days: int,
    ) -> float:
        """Calculate Value at Risk (VaR) using the historical simulation method.

        Args:
            returns (pandas.Series): A time series of historical returns.
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
            portfolio_value (int | float): The current value of the portfolio.
            historical_days (int): Number of historical days to consider for the simulation.
            horizon_days (int): Time horizon for VaR calculation in days.

        Returns:
            float: The calculated Value at Risk (VaR) for the portfolio over the given time horizon and confidence level.
        """
        first_return = max(len(returns) - historical_days, 0)
        returns_subset: pandas.Series = returns[first_return:]
        sorted_returns = numpy.sort(returns_subset)
        percentile = 1 - confidence_level
        percentile_sample_index = int(percentile * len(sorted_returns))
        worst_portfolio_value = sorted_returns[percentile_sample_index] * portfolio_value
        var = (portfolio_value - worst_portfolio_value) * numpy.sqrt(horizon_days)
        return var

    def _linear_model_var(
        self,
        returns: pandas.Series,
        confidence_level: float,
        portfolio_value: int | float,
        historical_days: int,
        horizon_days: int,
    ) -> float:
        """Calculate Value at Risk (VaR) using a linear model simulation.

        Args:
            returns (pandas.Series): A time series of historical returns.
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
            portfolio_value (int | float): The current value of the portfolio.
            historical_days (int): Number of historical days to consider for the simulation.
            horizon_days (int): Time horizon for VaR calculation in days.

        Returns:
            float: The calculated Value at Risk (VaR) for the portfolio over the given time horizon and confidence level.
        """
        first_return = max(len(returns) - historical_days, 0)
        returns_subset: pandas.Series = returns[first_return:]
        # Standard deviation is the statistical measure of market volatility
        std_dev = numpy.std(returns_subset)
        standard_score = norm.ppf(confidence_level)
        var = standard_score * std_dev * portfolio_value * numpy.sqrt(horizon_days)
        return var

    def _monte_carlo_var(
        self,
        returns: pandas.Series,
        confidence_level: float,
        portfolio_value: int | float,
        historical_days: int,
        horizon_days: int,
        number_of_samples: int = 5000,
    ) -> float:
        """Calculate Value at Risk (VaR) using a Monte Carlo simulation.

        Args:
            returns (pandas.Series): A time series of historical returns.
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
            portfolio_value (int | float): The current value of the portfolio.
            historical_days (int): Number of historical days to consider for the simulation.
            horizon_days (int): Time horizon for VaR calculation in days.
            number_of_samples (int, optional): Number of Monte Carlo simulation samples to generate. Defaults to 5000.

        Returns:
            float: The calculated Value at Risk (VaR) for the portfolio over the given time horizon and confidence level.
        """
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

    def _plot_hurst_eponent(self, intervals: list, data: list) -> float:
        """
        Plot the Hurst exponent on a log-log scale and calculate its value.

        This function creates a log-log plot of the mean rescaled range against interval size and calculates the Hurst exponent
        as the slope of the best-fit line.

        Args:
            intervals (list): List of interval sizes.
            data (list): List of mean rescaled ranges corresponding to the interval sizes.

        Returns:
            float: The calculated Hurst exponent.
            str: The base64-encoded PNG image of the plot.
        """
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
