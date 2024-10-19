# src/data_preprocessing/cleaner.py

import polars as pl
from polars import StringCache
import pandas as pd
import numpy as np

class PolarsLoader():
    def __init__(self, sampling=False):
        """
        Initializes the PolarsLoader class.

        Parameters:
            sampling (bool): If True, loads a sample of 1,000,000 rows from the dataset. Defaults to False.
        """
        self.sampling = sampling
    
    def load_data(self, path):
        
        """
        Loads the data from a CSV file.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            polars.DataFrame: Loaded DataFrame without selected columns.
        """
        if self.sampling:
            N_ROWS = 1_000_000
        else:
            N_ROWS = None

        # Read dataset as polars DataFrame
        df = pl.read_csv(path, low_memory=True,
                         batch_size=100_000, 
                         n_rows=N_ROWS
                         )
        
        # Drop not informative columns - according to EDA
        to_drop = [
            'ifa',
            'sdk',
            'adt',
            'dc',
            'ssp'
        ]

        df = (
                df.select([
                    pl.col('*').exclude(to_drop)
                ])
                .with_columns([ # Reduce string values in "bundle"
                    pl.col('bundle').cast(pl.Utf8).str.slice(12, 20)
                ])
            )

        return df 
    
    def downsample_data_pl(self, df: pl.DataFrame, neg_ratio: float = None, is_train: bool = True) -> pl.DataFrame:
        
        """
        Downsample the negative class of a DataFrame if is_train is True.

        Parameters:
            df (polars.DataFrame): DataFrame to be downsampled.
            neg_ratio (float): Ratio of negative samples to positive samples. If None, all negative samples are used. Defaults to None.
            is_train (bool): If True, downsamples data. Defaults to True.

        Returns:
            polars.DataFrame: Downsampled DataFrame if is_train is True.
        """
        if is_train:
            # Extract the counts of positive and negative cases
            p_cases = df.filter(pl.col('target') == 1)
            n_cases = df.filter(pl.col('target') == 0)

            # If neg_ratio is None use all negative samples
            if neg_ratio is not None:
                N = int(p_cases.height * neg_ratio)
                n_cases = n_cases.sample(n=N, seed=23)
            
            # Concatenar los casos negativos y positivos
            df = pl.concat([n_cases, p_cases])
            
        return df
    
    def set_datatypes(self, df):
        """
        Defines the data types of the columns in the DataFrame.
        Handles NA values in categorical columns.

        Parameters:
            df (polars.DataFrame): DataFrame to set the data types.

        Returns:
            df (polars.DataFrame): DataFrame with the data types set.
            cat_cols (list): List of categorical columns.
            num_cols (list): List of numerical columns.
        """

        # Define columns types
        # Float columns
        float_cols = [
            'flr',
            'sellerClearPrice',
            'price'
        ]

        # Integer columns
        int_cols = [
            'target',
            'hour',
            # 'ssp',
            'dsp',
            'request_context_device_w',
            'request_context_device_h',
            # 'contype', --> moved to cat_cols
            # 'request_context_device_type' --> moved to cat_cols
        ]

        # Categorical columns
        cat_cols = [
            'auctionBidFloorSource',
            'sdkver',
            'bundle',
            'os',
            'lang',
            'country',
            'region',
            'bidderFlrPolicy',
            'request_context_device_type',
            'contype',
            # 'sdk',
            # 'adt',
            # 'dc'
        ]

        # Set dtype for numeric columns (int)
        df = df.with_columns([pl.col(col).cast(pl.Int16) for col in int_cols])
                                    
        # Set dtype for numeric columns (float)
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
                
        # Set dtype for categorical columns
        df = df.with_columns([pl.col(col).cast(pl.String).cast(pl.Categorical) for col in cat_cols])

        # Handle NA values in categorical columns - Replace the value with "unknown"
        df = df.with_columns([pl.col(col).fill_null('unknown') for col in cat_cols])

        num_cols = float_cols + int_cols
        num_cols.remove('target')
        
        return df, cat_cols, num_cols
    
    def initial_preprocessing(self, df, neg_ratio=None, is_train=True):

        with StringCache():
            # Filter data
            df = self.load_data(df)

            # Downsample train data
            df = self.downsample_data_pl(df, neg_ratio=neg_ratio, is_train=is_train)
            
            # Set datatypes
            df, cat_cols, num_cols = self.set_datatypes(df)
        
        return df, cat_cols, num_cols