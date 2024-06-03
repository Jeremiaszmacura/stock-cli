from ports.stock_data_provider import StockDataProvider
from domain.plot_data import PlotTypes, plot_data
from domain.statistics import calculate_value_at_risk


def search_for_company_handler(stock_data_provider: StockDataProvider, search_phrase: str) -> None:
    """Handler used to search for a company by search phrase using specified stock data provider.

    Args:
        stock_data_provider (StockDataProvider): Selected stock data provider.
        search_phrase (str): Search phrase.
    """
    search_results: list[dict] = stock_data_provider.search_for_company(search_phrase)
    print(search_results)


def calculate_value_at_risk_handler(
    stock_data_provider: StockDataProvider,
    method: str,
    confidence_level: float,
    portfolio_value: int | float,
    historical_days: int,
    horizon_days: int,
) -> None:
    """Handler used to calculate Value at Risk statistic.

    Args:
        stock_data_provider (StockDataProvider): Selected stock data provider from avilable ones.
        method (str): Value at Risk calculation method.
        confidence_level (float): Confidence level.
        portfolio_value (int | float): Portfolio value.
        historical_days (int): Number of historical days for statistic calculation.
        horizon_days (int): Number of horizon days for statistic calculation.
    """
    stock_data = stock_data_provider.get_company_data()
    var: float = calculate_value_at_risk(
        method=method,
        data=stock_data,
        confidence_level=confidence_level,
        portfolio_value=portfolio_value,
        historical_days=historical_days,
        horizon_days=horizon_days,
    )
    print(var)


def draw_stock_graph_handler(stock_data_provider: StockDataProvider, plot_type: PlotTypes, company_symbol: str) -> None:
    """Handler used to draw stock graphs.

    Args:
        stock_data_provider (StockDataProvider): Selected stock data provider.
        plot_type (PlotTypes): Selected type of plot.
        company_symbol (str): Company stock symbol.
    """
    stock_data = stock_data_provider.get_company_data()
    plot_data(data=stock_data, plot_type=plot_type, company_symbol=company_symbol)
