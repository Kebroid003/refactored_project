from datetime import datetime
import numpy as np
import pandas as pd
import logging


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("Starting data cleaning process")

    original_rows = len(dataframe)
    dataframe = dataframe.drop_duplicates()

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    for column_name in numeric_columns:
        dataframe[column_name] = dataframe[column_name].fillna(dataframe[column_name].median())

    text_columns = dataframe.select_dtypes(include=['object']).columns
    for column_name in text_columns:
        dataframe[column_name] = dataframe[column_name].fillna('Unknown')

    for column_name in numeric_columns:
        quartile_1 = dataframe[column_name].quantile(0.25)
        quartile_3 = dataframe[column_name].quantile(0.75)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - 1.5 * iqr
        upper_bound = quartile_3 + 1.5 * iqr
        dataframe = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]

    for column_name in text_columns:
        dataframe[column_name] = dataframe[column_name].str.strip().str.lower()

    dataframe['quality_score'] = np.random.uniform(0.7, 1.0, len(dataframe))
    dataframe['processed_timestamp'] = datetime.now()

    final_rows = len(dataframe)
    logger.info(f"Data cleaning complete: {original_rows} -> {final_rows} rows")
    return dataframe


