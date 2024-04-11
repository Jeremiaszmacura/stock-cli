import logging
import typer
import keyring
import getpass
from typing_extensions import Annotated
from enum import Enum

from helpers.logger import logger
from domain.command_handler import search_for_company_handler, draw_stock_graph_handler
from domain.plot_data import PlotTypes
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
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
):
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
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
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
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Drawing stock graph for comapny: {company_symbol}")
    api_key: str = keyring.get_password(stock_data_provider.value, APP_NAME)
    if stock_data_provider == AvailableStockDataProviders.alpha_vantage:
        selected_stock_data_provider = AlphaVantage(auth_token=api_key, company_symbol=company_symbol)
    draw_stock_graph_handler(selected_stock_data_provider, plot_type)


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
        ),
    ] = AvailableStockDataProviders.alpha_vantage,
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Calculating value at risk for: {company_symbol}")


if __name__ == "__main__":
    app()
