from ports.stock_data_provider import StockDataProvider


def search_for_company_handler(stock_data_provider: StockDataProvider, search_phrase: str) -> None:
    search_results: list[dict] = stock_data_provider.search_for_company(search_phrase)
    print(search_results)


def draw_stock_graph_handler(stock_data_provider: StockDataProvider) -> None:
    pass
