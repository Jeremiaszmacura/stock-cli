import logging
import typer
import keyring
import getpass
from typing_extensions import Annotated
from enum import Enum

from helpers.logger import logger
from domain.command_handler import search_for_company_handler, draw_stock_graph_handler, calculate_value_at_risk_handler
from domain.plot_data import PlotTypes
from domain.statistics import ValueAtRiskMethods
from adapters.alpha_vantage_adapter import AlphaVantage


APP_NAME = "stock_hexagonal_app"


class AvailableStockDataProviders(str, Enum):
    alpha_vantage = "alpha_vantage"


app = typer.Typer(help="CLI Application for analyzing stock data.")

statistics_app = typer.Typer()
app.add_typer(statistics_app, name="count-statistics", help="Calculate selected statistics.")


@app.command()
def store_api_key(
    stock_data_provider: Annotated[
        AvailableStockDataProviders,
        typer.Option(
            "--stock-data-provider",
            "-stp",
            case_sensitive=False,
            help="Selected stock data provider from avilable ones.",
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
):
    """CLI command used to store auth token / api key to be reused in other CLI commands.

    Args:
        stock_data_provider (AvailableStockDataProviders): Stock data provider.
    """
    api_key: str = getpass.getpass(prompt="Provide your API Key:\n")
    keyring.set_password(stock_data_provider.value, APP_NAME, api_key)


@app.command()
def search_for_company(
    phrase: Annotated[
        str,
        typer.Option(
            "--phrase",
            "-p",
            help="Search for company by passed phrase.",
        ),
    ],
    stock_data_provider: Annotated[
        AvailableStockDataProviders,
        typer.Option(
            "--stock-data-provider",
            "-stp",
            case_sensitive=False,
            help="Selected stock data provider from avilable ones.",
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    """CLI command used to search for a company by provided phrase.

    Args:
        phrase (str): phrase used to search for copmany.
        stock_data_provider (AvailableStockDataProviders): Stock data provider.
        debug (bool): Switch logging debug mode.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("Searching for comapny by phrase: {phrase}")
    api_key: str = keyring.get_password(stock_data_provider.value, APP_NAME)
    if stock_data_provider == AvailableStockDataProviders.alpha_vantage:
        selected_stock_data_provider = AlphaVantage(auth_token=api_key)
    search_for_company_handler(selected_stock_data_provider, phrase)


@app.command()
def draw_stock_graph(
    company_symbol: Annotated[
        str,
        typer.Option(
            "--company-symbol",
            "-s",
            help="Company stock symbol.",
        ),
    ],
    stock_data_provider: Annotated[
        AvailableStockDataProviders,
        typer.Option(
            "--stock-data-provider",
            "-stp",
            case_sensitive=False,
            help="Selected stock data provider from avilable ones.",
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
    plot_type: Annotated[
        PlotTypes,
        typer.Option(
            case_sensitive=False,
        ),
    ] = PlotTypes.linear_plot,
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    """CLI command used to plot chart for selected company.

    Args:
        company_symbol (str): Company stock symbol.
        stock_data_provider (AvailableStockDataProviders): Stock data provider.
        plot_type (PlotTypes): Plot type.
        debug (bool): Switch logging debug mode.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Drawing stock graph for comapny: {company_symbol}")
    api_key: str = keyring.get_password(stock_data_provider.value, APP_NAME)
    if stock_data_provider == AvailableStockDataProviders.alpha_vantage:
        selected_stock_data_provider = AlphaVantage(auth_token=api_key, company_symbol=company_symbol)
    draw_stock_graph_handler(selected_stock_data_provider, plot_type, company_symbol)


@statistics_app.command()
def value_at_risk(
    company_symbol: Annotated[
        str,
        typer.Option(
            "--company-symbol",
            "-s",
            help="Company stock symbol.",
        ),
    ],
    stock_data_provider: Annotated[
        AvailableStockDataProviders,
        typer.Option(
            "--stock-data-provider",
            "-stp",
            case_sensitive=False,
            help="Selected stock data provider from avilable ones.",
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
    method: Annotated[
        ValueAtRiskMethods,
        typer.Option(
            "--method",
            "-m",
            case_sensitive=False,
            help="Value at Risk calculation method.",
        ),
    ] = ValueAtRiskMethods.historical_simulation,
    confidence_level: Annotated[
        float,
        typer.Option(
            "--confidence-level",
            "-cl",
            help="Confidence level.",
        ),
    ] = 0.99,
    portfolio_value: Annotated[
        float,
        typer.Option(
            "--portfolio-value",
            "-pv",
            help="Portfolio value.",
        ),
    ] = 100,
    historical_days: Annotated[
        int,
        typer.Option(
            "--historical-days",
            "-hd",
            help="Number of historical days for statistic calculation.",
        ),
    ] = 200,
    horizon_days: Annotated[
        int,
        typer.Option(
            "--horizon-days",
            "-hod",
            help="Number of horizon days for statistic calculation.",
        ),
    ] = 1,
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    """CLI command used to calculate Value at Risk for selected company.

    Args:
        company_symbol (str): Company stock symbol.
        stock_data_provider (AvailableStockDataProviders): Stock data provider.
        method (ValueAtRiskMethods): Value at Risk calculation method.
        debug (bool): Switch logging debug mode.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Calculating value at risk for: {company_symbol}")

    api_key: str = keyring.get_password(stock_data_provider.value, APP_NAME)
    if stock_data_provider == AvailableStockDataProviders.alpha_vantage:
        selected_stock_data_provider = AlphaVantage(auth_token=api_key)
    calculate_value_at_risk_handler(selected_stock_data_provider, method)


if __name__ == "__main__":
    app()
