import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def generate_visualizations(dataframe: pd.DataFrame, output_dir: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations")
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8')

    numeric_df = dataframe.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    numeric_columns = numeric_df.columns[:6]
    if len(numeric_columns) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for i, col in enumerate(numeric_columns):
            if i < 6:
                axes[i].hist(dataframe[col].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    if len(numeric_columns) > 0:
        plt.figure(figsize=(12, 6))
        dataframe[numeric_columns].boxplot()
        plt.title('Box Plots for Outlier Detection')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


