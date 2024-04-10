from ports.stock_data_provider import StockDataProvider


def search_for_company_handler(stock_data_provider: StockDataProvider, search_phrase: str) -> None:
    stock_data_provider.search_for_company(search_phrase)
