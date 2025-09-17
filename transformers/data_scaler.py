import numpy as np
import pandas as pd
import logging


def transform_data(dataframe: pd.DataFrame, transformation_type: str = 'standard') -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info(f"Applying {transformation_type} transformation")

    try:
        if transformation_type == 'standard':
            numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
            for column_name in numeric_columns:
                if column_name not in ['quality_score']:
                    mean_val = dataframe[column_name].mean()
                    std_val = dataframe[column_name].std()
                    if std_val != 0:
                        dataframe[f'{column_name}_standardized'] = (dataframe[column_name] - mean_val) / std_val

        elif transformation_type == 'normalize':
            numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
            for column_name in numeric_columns:
                if column_name not in ['quality_score']:
                    min_val = dataframe[column_name].min()
                    max_val = dataframe[column_name].max()
                    if max_val != min_val:
                        dataframe[f'{column_name}_normalized'] = (dataframe[column_name] - min_val) / (max_val - min_val)

        elif transformation_type == 'categorical':
            text_columns = dataframe.select_dtypes(include=['object']).columns
            for column_name in text_columns:
                if column_name not in ['processed_timestamp']:
                    dataframe[f'{column_name}_encoded'] = pd.Categorical(dataframe[column_name]).codes

        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            dataframe['feature_sum'] = dataframe[numeric_columns].sum(axis=1)
            dataframe['feature_mean'] = dataframe[numeric_columns].mean(axis=1)
            dataframe['feature_std'] = dataframe[numeric_columns].std(axis=1)

        logger.info(f"Transformation complete: {len(dataframe.columns)} columns")
        return dataframe
    except Exception:
        logger.exception("Data transformation failed")
        return dataframe


