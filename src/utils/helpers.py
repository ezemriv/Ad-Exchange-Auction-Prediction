import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from imblearn.pipeline import Pipeline
import os

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

def replace_less_frequent_polars(df: pl.DataFrame, list_col: list[str], 
                                 threshold: float = 0.05, new_value='other', 
                                 replace_values: dict = None) -> tuple[pl.DataFrame, dict]:

    """
    NEEDS POLARS StringCache()!!

    Replaces less frequent values in specified columns of a DataFrame with a new value.
    If replace_values is provided, it uses that list of values to replace in the columns.
    """
    if replace_values is None:
        replace_values = {}

    for col in list_col:
        # If replace_values are not provided, calculate them from the DataFrame
        if col not in replace_values:
            vals_to_replace = (
                df[col].value_counts(normalize=True, sort=True)
                .filter(pl.col("proportion") < threshold)[col]
                .to_list()
            )
            replace_values[col] = vals_to_replace  # Store these values for later use (train set)
        else:
            vals_to_replace = replace_values[col]  # Use pre-defined values for replacement (test set)

        # Replace less frequent values with `new_value`
        df = df.with_columns([
                pl.when(pl.col(col).is_in(vals_to_replace))
                .then(pl.lit(new_value))
                .otherwise(pl.col(col))
                .alias(col)
                .cast(pl.Categorical)
        ])

    return df, replace_values  # Return the modified DataFrame and the replace values for reuse
    
def plot_cross_validation_scores(train_scores, valid_scores, results_folder):
    """Plots and saves the cross-validation scores."""
    plt.figure()
    plt.plot(range(1, len(train_scores) + 1), train_scores,
             marker='o', label='Train F1 Score', linestyle='-')
    plt.plot(range(1, len(valid_scores) + 1), valid_scores,
             marker='o', label='Validation F1 Score', linestyle='-')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Train and Validation F1 Scores')
    plt.legend()
    cv_plot_path = os.path.join(results_folder, 'cross_validation_scores.png')
    plt.savefig(cv_plot_path)
    plt.close()
    print(f"Cross-validation plot saved to {cv_plot_path}")

def plot_feature_importances(model, FEATURES, results_folder):
    """Plots and saves the feature importances."""
    # Handle pipeline if oversampling is used
    if isinstance(model, Pipeline):
        classifier = model.named_steps['classifier']
    else:
        classifier = model

    # Get feature importances
    importances = classifier.feature_importances_
    importances_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title('Feature Importances')
    fi_plot_path = os.path.join(results_folder, 'feature_importances.png')
    plt.savefig(fi_plot_path)
    plt.close()
    print(f"Feature importance plot saved to {fi_plot_path}")