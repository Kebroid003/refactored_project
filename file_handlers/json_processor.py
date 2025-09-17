import os
import json
import pandas as pd
from typing import Optional
import logging

from utils.file_validator import validate_file, calculate_file_hash
from database.schemas import insert_raw_data_log


def load_json_file(file_path: str, connection) -> Optional[pd.DataFrame]:
    logging.getLogger(__name__).info(f"Loading JSON file: {file_path}")

    if not validate_file(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        dataframe = pd.DataFrame(data)
    elif isinstance(data, dict):
        dataframe = pd.DataFrame([data])
    else:
        raise ValueError("Unsupported JSON structure")

    file_hash = calculate_file_hash(file_path)
    insert_raw_data_log(connection, file_path, file_hash or '', len(dataframe), os.path.getsize(file_path))

    logging.getLogger(__name__).info(f"Successfully loaded {len(dataframe)} rows from JSON")
    return dataframe


