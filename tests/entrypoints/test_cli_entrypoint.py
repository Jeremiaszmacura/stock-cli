import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.entrypoints.cli_entrypoint import app


runner = CliRunner()


class TestSearchForCompany(unittest.TestCase):
    @patch("keyring.get_password")
    def test_search_for_company(self, get_password):
        get_password.return_value = MagicMock()
        result = runner.invoke(app, ["search-for-company", "--phrase", "INTC"])
        assert result.exit_code == 0
        get_password.assert_called_once()
