from ports.stock_data_provider import StockDataProvider
from domain.plot_data import PlotTypes, plot_data


def search_for_company_handler(stock_data_provider: StockDataProvider, search_phrase: str) -> None:
    search_results: list[dict] = stock_data_provider.search_for_company(search_phrase)
    print(search_results)


def draw_stock_graph_handler(stock_data_provider: StockDataProvider, plot_type: PlotTypes) -> None:
    stock_data = stock_data_provider.get_company_data()
    plot_data(data=stock_data, plot_type=plot_type)
