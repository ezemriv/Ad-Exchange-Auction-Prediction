# src/data_preprocessing/feature_engineering.py

import polars as pl

class FeatureEngineering:
    
    def __init__(self):
        pass
    
    def feature_engineering(self, df):
        
        df = df.with_columns([
            pl.col('bundle').to_physical().cast(pl.String).alias("bundle_int").cast(pl.Categorical),
        ])
        
        df = df.drop('bundle')
        
        return df

    def grouped_feature_engineering(self, df):
            
                    
            # df = df.with_columns([     
                
            #     # Ratio of tbp_lv_contrast_A to age average
            #     pl.col('tbp_lv_contrast_A').truediv(pl.col('tbp_lv_contrast_A').mean())
            #     .over('age_approx')
            #     .cast(pl.Float32).alias('tbp_lv_age_contrast_A'),
                
            # ])
            
           
            return df
  
    def downsample_data_pl(self, df: pl.DataFrame, neg_ratio: float = None, is_train: bool = True) -> pl.DataFrame:
        
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

    def process_data(self, df, neg_ratio=None, is_train=True, img_fea_path=None):

        if DOMAIN_FEATURES:
            # Create initial features
            df = self.feature_engineering(df)

        # # Check if group-specific features should be added
        # if GROUP_FEATURES:
        #     # Create age and patient group features
        #     df = self.grouped_feature_engineering(df)

        # Downsample data if image features are not used
        if DOWNSAMPLE:
            df = self.downsample_data_pl(df, neg_ratio, is_train)

        # Convert to pandas DataFrame
        df = df.to_pandas()

        return df
