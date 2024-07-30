"""Module contain abstract class for stock data provider."""

from abc import ABC, abstractmethod
import pandas


class StockDataProvider(ABC):
    """Stock data provider port."""

    auth_token: str
    company_symbol: str | None = None
    time_interval: str | None = None

    @abstractmethod
    def get_company_data(self) -> pandas.DataFrame:
        """Property used to get and keep company stock data."""

    @abstractmethod
    def search_for_company(self, search_phrase: str) -> list[dict]:
        """Method to search for company based on search phrase.

        Args:
            search_phrase (str): Phrase used to search for company.
        """

    @abstractmethod
    def _prepare_search_result_data(self, data: dict) -> list[dict]:
        """Method to prepare search result data to display it in a target format.

        Args:
            data dict: Search result data in stock data provider format.

        Returns:
            list[dict]: Search result data in target format.
        """

    @abstractmethod
    def _prepare_data(self, data: bytes) -> pandas.DataFrame:
        """Method to prepare company's stock data for futher calculations.

        Args:
            data (bytes): Company's stock data in stock data provider format.

        Returns:
            pandas.DataFrame: Company's stock data in target format.
        """
