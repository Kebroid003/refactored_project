import os
import pandas as pd
from typing import Optional
import logging

from utils.file_validator import validate_file, calculate_file_hash
from database.schemas import insert_raw_data_log


def load_csv_file(file_path: str, chunk_size: int, connection) -> Optional[pd.DataFrame]:
    logging.getLogger(__name__).info(f"Loading CSV file: {file_path}")

    if not validate_file(file_path):
        return None

    encodings = ['utf-8', 'latin1', 'cp1252']
    dataframe_iterator = None
    for encoding in encodings:
        try:
            dataframe_iterator = pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size)
            break
        except UnicodeDecodeError:
            continue

    if dataframe_iterator is None:
        logging.getLogger(__name__).error("Could not decode file with any encoding")
        return None

    all_chunks = [chunk for chunk in dataframe_iterator]
    final_df = pd.concat(all_chunks, ignore_index=True)

    file_hash = calculate_file_hash(file_path)
    insert_raw_data_log(connection, file_path, file_hash or '', len(final_df), os.path.getsize(file_path))

    logging.getLogger(__name__).info(f"Successfully loaded {len(final_df)} rows from {file_path}")
    return final_df


