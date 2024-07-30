import requests
import pandas
import io

from ports.stock_data_provider import StockDataProvider


class AlphaVantage(StockDataProvider):
    def __init__(
        self,
        auth_token: str,
        company_symbol: str | None = None,
        time_interval: str | None = None,
    ):
        self.auth_token = auth_token
        self.company_symbol = company_symbol
        self.time_interval = time_interval

    def get_company_data(self) -> pandas.DataFrame:
        """Request for stock data from external data provider/vendor.

        Returns:
            pandas.DataFrame: prepared stock data for futher calculations.
        """
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.company_symbol}&apikey={self.auth_token}&datatype=csv"
        res: requests.models.Response = requests.get(url, timeout=10)
        data_bytes: bytes = res.content
        prepared_data = self._prepare_data(data_bytes)
        return prepared_data

    def search_for_company(self, search_phrase: str) -> list[dict]:
        """Method used to search for company using passed serach phrase.

        Args:
            search_phrase (str): phrase used to search for company.

        Returns:
            list[dict]: List of top best comopany matches for passed phrase.
        """
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={search_phrase}&apikey={self.auth_token}"
        res: requests.models.Response = requests.get(url, timeout=10)
        data: dict = res.json()
        search_result: list[dict] = self._prepare_search_result_data(data)
        return search_result

    def _prepare_search_result_data(self, data: dict) -> list[dict]:
        """Method prepares search result data to be displayed for user.

        Args:
            data (dict): search result data.

        Returns:
            list[dict]: prepared data to be displayed for user.
        """
        search_result: list[dict] = data["bestMatches"]
        return search_result

    def _prepare_data(self, data: bytes) -> pandas.DataFrame:
        """Method prepares company stock data for futher calculations.

        Args:
            data (bytes): company stock data.

        Returns:
            pandas.DataFrame: prepared company stock data.
        """
        data_str: str = data.decode()
        data_file: io.StringIO = io.StringIO(data_str)
        prepared_data: pandas.DataFrame = pandas.read_csv(data_file)
        return prepared_data
