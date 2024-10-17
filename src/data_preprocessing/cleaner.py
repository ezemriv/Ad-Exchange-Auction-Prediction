# src/data_preprocessing/cleaner.py

import polars as pl
from polars import StringCache

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

        for col in to_drop:
            if col in df.columns:
                df = df.drop(col)

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

        for col in int_cols:
            
            # Set dtype for numeric columns (int)
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
                        
        for col in float_cols:
            
            # Set dtype for numeric columns (float)
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
                
        for col in cat_cols:
            
            # Set dtype for categorical columns
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))

            # Handle NA values in categorical columns - Replace the value with "unknown"
                df = df.with_columns(pl.col(col).fill_null('unknown'))

        num_cols = float_cols + int_cols
        num_cols.remove('target')
        
        return df, cat_cols, num_cols
    
    def initial_preprocessing(self, df):

        with StringCache():
            # Filter data
            df = self.load_data(df)
            
            # Set datatypes
            df, cat_cols, num_cols = self.set_datatypes(df)
        
        return df, cat_cols, num_cols
    
    def save_clean_data(self, df, filepath):
        
        df.write_parquet(filepath)
        print(f"Cleaned data saved at {filepath}")
