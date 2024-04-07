"""Module designed to init and configure logger."""
import logging
import datetime


logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"loggs/{datetime.datetime.now()}-stock-app.log", level=logging.INFO)
