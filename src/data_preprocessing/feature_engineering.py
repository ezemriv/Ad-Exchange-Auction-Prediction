import polars as pl
from polars import StringCache
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from src.utils.helpers import replace_less_frequent_polars

class FeatureEngineering:
    
    def __init__(self):
        self.category_encoder = None
        self.low_freq_values = {}  # Stores low-frequency values identified during training
    
    def feature_engineering(self, df, is_train=True):

        float_cols = ['flr', 'sellerClearPrice', 'price']

        # Apply log transformation to price columns
        df = df.with_columns([
            (pl.col(col) + 1).log().alias(f"{col}_log")
            for col in float_cols
            if col in df.columns
        ])
        
        # Add categorical hour
        df = df.with_columns([
            pl.col("hour").cast(pl.String).cast(pl.Categorical).alias("hour_cat")
        ])

        def categorize_hour(hour):
            if hour == 0:
                return "Night"
            elif 1 <= hour <= 6:
                return "Late Night"
            elif 7 <= hour <= 12:
                return "Morning"
            elif 13 <= hour <= 18:
                return "Afternoon"
            elif 19 <= hour <= 23:
                return "Night"
            else:
                return "Invalid hour"

        df = df.with_columns([
            pl.col("hour").map_elements(categorize_hour, return_dtype=pl.String).cast(pl.Categorical).alias("hour_band")
        ])
    
        # Extract general language from lang col
        df = df.with_columns([
            pl.col("lang")
            .cast(pl.String)
            .str.split_exact("_", 1)  
            .struct.field("field_0")  # Extract only the "first_part"
            .alias("lang")
            .cast(pl.Categorical)
        ])

        # Screen width and height
        df = df.with_columns([
            (pl.col("request_context_device_w").cast(pl.Utf8) + "x" + pl.col("request_context_device_h").cast(pl.Utf8))
            .alias("screen_size")
            .cast(pl.Categorical)
        ])

        # Group low frequency categories into "other"
        low_freq = ['sdkver', 'lang', 'country', 'region', 'screen_size']

        # If it's training data, find and store the low-frequency values; if test, reuse stored values
        if is_train:
            df, self.low_freq_values = replace_less_frequent_polars(df, low_freq, threshold=0.01, replace_values=None)
        else:
            df, _ = replace_less_frequent_polars(df, low_freq, threshold=0.01, replace_values=self.low_freq_values)

        new_num_cols = [
                    'flr_log',
                    'sellerClearPrice_log',
                    'price_log',
                    ]

        new_cat_cols = [
                    'hour_cat',
                    'hour_band',
                    'screen_size'
                    ]

        new_cols = new_num_cols + new_cat_cols

        return df, new_cols, new_cat_cols

    def grouped_feature_engineering(self, df):

        float_cols = ['flr', 'sellerClearPrice', 'price']

        df = df.with_columns([
                        ((pl.col(col) - pl.col(col).mean().over('country')) / (pl.col(col).std().over('country') + 1e-6)).
                        alias(f'{col}_country_norm') for col in float_cols
                ])
        
        df = df.with_columns([
                        ((pl.col(col) - pl.col(col).mean().over('hour_band')) / (pl.col(col).std().over('hour_band') + 1e-6)).
                        alias(f'{col}_hour_band_norm') for col in float_cols
                ])
        
        df = df.with_columns([
                        ((pl.col(col) - pl.col(col).mean().over('auctionBidFloorSource')) / (pl.col(col).std().over('auctionBidFloorSource') + 1e-6)).
                        alias(f'{col}_auctionBidFloorSource_norm') for col in float_cols
                ])       

        norm_cols = (
            [col + '_country_norm' for col in float_cols] + 
            [col + '_hour_band_norm' for col in float_cols] + 
            [col + '_auctionBidFloorSource_norm' for col in float_cols]
        )

        return df, norm_cols
  
    def encode_categoricals(self, df, cat_cols, is_train=True):
        """
        Encodes categorical columns using OrdinalEncoder.
        
        Parameters:
        - df (pl.DataFrame): The input DataFrame.
        - cat_cols (list): List of categorical column names to encode.
        - is_train (bool): Flag indicating whether the data is training data.
        
        Returns:
        - pl.DataFrame: The DataFrame with encoded categorical columns.
        """
        if is_train:
            # Initialize and fit the encoder on training data
            self.category_encoder = OrdinalEncoder(
                categories='auto', dtype=np.int16, 
                handle_unknown='use_encoded_value', 
                unknown_value=-2, encoded_missing_value=-1
            )
            # Fit and transform the category columns
            data_encoded = self.category_encoder.fit_transform(df.select(cat_cols))
        else:
            # Ensure the encoder has been fitted
            if self.category_encoder is None:
                raise ValueError("The encoder has not been fitted. Call encode_categoricals with is_train=True first.")
            # Transform the test data with the same encoder
            data_encoded = self.category_encoder.transform(df.select(cat_cols))
        
        # Assign the transformed categories back to the Polars DataFrame
        encoded_columns = [
            pl.Series(cat_col, data_encoded[:, idx]) 
            for idx, cat_col in enumerate(cat_cols)
        ]
        df = df.with_columns(encoded_columns)
        
        # Cast all the cat_cols back to categorical
        df = df.with_columns([
            pl.col(col).cast(pl.String).cast(pl.Categorical) for col in cat_cols
        ])
        
        return df
    
    def proccess(self, df, cat_cols, is_train=True):

        with StringCache():

            # Feature engineering
            df, new_cols, new_cat_cols = self.feature_engineering(df, is_train)
            cat_cols += new_cat_cols

            df, norm_cols = self.grouped_feature_engineering(df)
            new_cols += norm_cols

            # df = self.encode_categoricals(df, cat_cols, is_train) # -> same result. LightGBM handles them automatically

        # Convert to pandas dataframe
        df = df.to_pandas()

        return df, new_cols