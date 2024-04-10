# Python CLI tool to analyze stock data

## How to use

```sh
pip install stock-hexagonal-app
stock-hexagonal-app --help
```

## Hexagonal Architecture

This tool is build using hexagonal architecture with following structure

    |--- src/  # package source code
        |--- adapters/  # implementation of the ports defined in the domain
        |--- domain/  # implementation of business logic
        |--- entrypoints/  # primary adapters, entry points
        |--- ports/  # abstractions for adapters
        |--- helpers/  # modules containing helpers functions, configurations
    |--- tests/  # package tests
