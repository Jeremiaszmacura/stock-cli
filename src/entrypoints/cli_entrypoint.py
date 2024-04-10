import logging
import typer
from typing_extensions import Annotated
from enum import Enum

from helpers.logger import logger
from domain.command_handler import search_for_company_handler
from adapters.alpha_vantage_adapter import AlphaVantage


class AvailableStockDataProviders(str, Enum):
    alpha_vantage = "alpha_vantage"


app = typer.Typer(help="CLI Application for analyzing stock data.")

statistics_app = typer.Typer()
app.add_typer(statistics_app, name="count-statistics", help="Calculate selected statistics.")


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
    print(f"Searching for comapny by phrase: {phrase}")
    if stock_data_provider == AvailableStockDataProviders.alpha_vantage:
        selected_stock_data_provider = AlphaVantage()
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
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    if debug:
        logger.setLevel(logging.DEBUG)
    print(f"Drawing stock graph for comapny: {company_symbol}")


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
    debug: Annotated[bool, typer.Option(help="Switch logger debug mode.")] = False,
):
    if debug:
        logger.setLevel(logging.DEBUG)
    print(f"Calculating value at risk for: {company_symbol}")


if __name__ == "__main__":
    app()
