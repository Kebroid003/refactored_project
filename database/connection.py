import os
import sqlite3
import logging


def get_connection(database_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    logging.getLogger(__name__).info("Database connection established")
    return connection


