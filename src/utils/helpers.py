import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

class InitialEDA:
    """Class for performing exploratory data analysis (EDA) on a DataFrame."""

    @staticmethod
    def plot_histograms(df, num_cols: list[str] = []):
        """Plots histograms for numeric variables in the DataFrame."""
        df[num_cols].hist(bins=15, figsize=(15, 10))
        plt.suptitle('Histograms of Numeric Variables')
        plt.show()
    
    @staticmethod
    def plot_barplots_normalized(df, exclude: list[str] = [], cat_cols: list[str] = []):
        """Plots horizontal bar plots of normalized value counts for categorical variables in the DataFrame."""
        
        categorical_cols = [col for col in cat_cols if col not in exclude]

        # Set number of rows and columns for subplots
        n_cols = 2
        n_rows = math.ceil(len(categorical_cols) / n_cols)
        
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, len(categorical_cols) * 3))
        axs = axs.flatten()  # Flatten in case we have multiple rows and columns

        for i, col in enumerate(categorical_cols):
            sns.barplot(x=df[col].value_counts(normalize=True), 
                        y=df[col].value_counts(normalize=True).index, 
                        ax=axs[i])
            axs[i].set_title(f'Normalized Value Counts of {col}')
            axs[i].set_xlabel('Normalized Value Counts')
            axs[i].set_ylabel('Categories')

        # Hide any extra axes
        for i in range(len(categorical_cols), len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

class Preprocessing():
    """Class for data preprocessing tasks."""

    @staticmethod
    def replace_less_frequent(df: pd.DataFrame, list_col: list[str], 
                              threshold: float = 0.02, new_value='other') -> pd.DataFrame:
        """Replaces less frequent values in specified columns of a DataFrame with a new value."""
        for col in list_col:
            freq = df[col].value_counts(normalize=True)
            to_replace = freq[freq < threshold].index
            df[col].replace(to_replace, new_value, inplace=True)
        return df