# main.py

import os
from src.data_preprocessing.cleaner import DataCleaner
from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.models.baseline_model import BaselineModel
from src.models.advanced_model import AdvancedModel
from src.evaluation.metrics import ModelEvaluator
import yaml

def main():
    # Load configuration
    with open('configs/model_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize data cleaner and feature engineer
    cleaner = DataCleaner()
    engineer = FeatureEngineer()

    # Load raw data
    raw_data_path = 'data/train_data.csv'
    df = None  # Replace with code to load data

    # Data cleaning
    df_clean = cleaner.handle_missing_values(df)
    df_clean = cleaner.remove_outliers(df_clean)
    cleaner.save_clean_data(df_clean, 'data/clean_data.csv')

    # Feature engineering
    df_features = engineer.encode_categorical_variables(df_clean)
    df_features = engineer.create_new_features(df_features)
    engineer.save_features(df_features, 'data/processed_data.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = None, None, None, None  # Replace with data splitting code

    # Initialize models
    baseline_model = BaselineModel(config['baseline_model'])
    advanced_model = AdvancedModel(config['advanced_model'])

    # Train and evaluate baseline model
    baseline_model.train_model(X_train, y_train)
    y_pred_baseline = baseline_model.evaluate_model(X_test, y_test)
    baseline_model.save_model('models/baseline_model.pkl')

    # Train and evaluate advanced model
    advanced_model.train_model(X_train, y_train)
    advanced_model.hyperparameter_tuning(X_train, y_train)
    y_pred_advanced = advanced_model.evaluate_model(X_test, y_test)
    advanced_model.save_model('models/advanced_model.pkl')

    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.calculate_f1_score(y_test, y_pred_baseline)
    evaluator.calculate_f1_score(y_test, y_pred_advanced)

    # Compare models and output results
    pass  # Replace with code to compare and output results

if __name__ == '__main__':
    main()
