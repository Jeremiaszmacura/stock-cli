from ports.stock_data_provider import StockDataProvider
from domain.plot_data import PlotTypes, plot_data


def search_for_company_handler(stock_data_provider: StockDataProvider, search_phrase: str) -> None:
    """Handler used to search for a company by search phrase using specified stock data provider.

    Args:
        stock_data_provider (StockDataProvider): Selected stock data provider.
        search_phrase (str): Search phrase.
    """
    search_results: list[dict] = stock_data_provider.search_for_company(search_phrase)
    print(search_results)


def draw_stock_graph_handler(stock_data_provider: StockDataProvider, plot_type: PlotTypes, company_symbol: str) -> None:
    """Handler used to draw stock graphs.

    Args:
        stock_data_provider (StockDataProvider): Selected stock data provider.
        plot_type (PlotTypes): Selected type of plot.
        company_symbol (str): Company stock symbol.
    """
    stock_data = stock_data_provider.get_company_data()
    plot_data(data=stock_data, plot_type=plot_type, company_symbol=company_symbol)
