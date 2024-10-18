import pandas as pd
import numpy as np
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