import os
from pathlib import Path
import pandas as pd

from config.settings import DEFAULT_CONFIG, overlay_config
from utils.logging_setup import setup_logging
from database.connection import get_connection
from database.schemas import create_tables
from file_handlers.csv_processor import load_csv_file
from file_handlers.json_processor import load_json_file
from transformers.data_cleaner import clean_data
from transformers.data_scaler import transform_data
from visualizations.chart_generator import generate_visualizations
from reports.generator import generate_text_summary, generate_detailed_html
from utils.backup_manager import backup_data


def load_config_from_ini(config_file_path: str) -> dict:
    import configparser
    parser = configparser.ConfigParser()
    if os.path.exists(config_file_path):
        parser.read(config_file_path)
        overrides: dict = {}
        for section in parser.sections():
            for key, value in parser.items(section):
                lowered = value.lower()
                if lowered in ['true', 'false']:
                    overrides[key] = parser.getboolean(section, key)
                elif value.isdigit():
                    overrides[key] = parser.getint(section, key)
                else:
                    try:
                        overrides[key] = float(value)
                    except ValueError:
                        overrides[key] = value
        return overlay_config(DEFAULT_CONFIG, overrides)
    return DEFAULT_CONFIG.copy()


def calculate_statistics(dataframe: pd.DataFrame) -> dict:
    import numpy as np
    stats: dict = {}
    stats['total_rows'] = len(dataframe)
    stats['total_columns'] = len(dataframe.columns)
    stats['memory_usage'] = dataframe.memory_usage(deep=True).sum()

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        stats[f'{col}_mean'] = dataframe[col].mean()
        stats[f'{col}_median'] = dataframe[col].median()
        stats[f'{col}_std'] = dataframe[col].std()
        stats[f'{col}_min'] = dataframe[col].min()
        stats[f'{col}_max'] = dataframe[col].max()
        stats[f'{col}_null_count'] = dataframe[col].isnull().sum()

    text_columns = dataframe.select_dtypes(include=['object']).columns
    for col in text_columns:
        stats[f'{col}_unique_count'] = dataframe[col].nunique()
        stats[f'{col}_most_common'] = dataframe[col].mode().iloc[0] if not dataframe[col].mode().empty else 'N/A'
        stats[f'{col}_null_count'] = dataframe[col].isnull().sum()

    stats['overall_quality_score'] = dataframe.get('quality_score', pd.Series([0])).mean()
    stats['completeness_ratio'] = 1 - (dataframe.isnull().sum().sum() / (len(dataframe) * len(dataframe.columns)))
    return stats


def process_directory(config: dict) -> pd.DataFrame | None:
    logger = setup_logging(config['log_file'])
    logger.info(f"Processing directory: {config['input_directory']}")

    connection = get_connection(config['database_path'])
    create_tables(connection)

    all_processed: list[pd.DataFrame] = []
    for file_path in Path(config['input_directory']).glob('*'):
        if not file_path.is_file():
            continue
        df: pd.DataFrame | None = None
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            df = load_csv_file(str(file_path), config['chunk_size'], connection)
        elif suffix == '.json':
            df = load_json_file(str(file_path), connection)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            continue

        if df is None:
            continue

        df = clean_data(df)
        df = transform_data(df)
        all_processed.append(df)

        os.makedirs(config['output_directory'], exist_ok=True)
        output_path = f"{config['output_directory']}/processed_{file_path.stem}.csv"
        df.to_csv(output_path, index=False)

    if not all_processed:
        return None

    combined_df = pd.concat(all_processed, ignore_index=True)
    stats = calculate_statistics(combined_df)

    viz_dir = f"{config['output_directory']}/visualizations"
    generate_visualizations(combined_df, viz_dir)

    generate_text_summary(stats, connection, config['reports_directory'])
    generate_detailed_html(stats, connection, config['reports_directory'])

    if config.get('backup_enabled', False):
        backup_data(config['database_path'], config['output_directory'], config['reports_directory'])

    # cleanup could be added here as a separate util later
    return combined_df


def main():
    config_file = 'config/settings.ini'
    config = load_config_from_ini(config_file)
    result = process_directory(config)
    if result is not None:
        print(f"Processing completed. {len(result)} total rows processed.")
    else:
        print("Processing failed or no files processed.")


if __name__ == "__main__":
    main()


