import pandas

from src.ports.stock_data_provider import StockDataProvider


class AlphaVantage(StockDataProvider):
    def __init__(
        auth_token: str,
        company_symbol: str | None = None,
        time_interval: str | None = None,
        company_data: pandas.DataFrame | None = None,
    ):
        auth_token = auth_token
        company_symbol = company_symbol
        time_interval = time_interval
        company_data = company_data

    def prepare_search_result_data(data: list[dict]) -> list[dict]:
        pass

    def prepare_data(data: pandas.DataFrame) -> pandas.DataFrame:
        pass

    def search_for_company(search_phrase: str) -> None:
        pass

    def get_company_data() -> None:
        pass
