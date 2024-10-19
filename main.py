# main.py

import sys
import os
import yaml
import pandas as pd
import numpy as np
import polars as pl
from polars import StringCache

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import helper functions
from src.data_preprocessing.cleaner import PolarsLoader
from src.data_preprocessing.feature_engineering import FeatureEngineering
from src.utils.helpers import plot_cross_validation_scores, plot_feature_importances

def load_config(config_file='configs/model_config.yaml'):
    """Loads configuration parameters from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_data(config):

    """Loads and preprocesses the training and test data."""
    # Set up data paths
    TRAIN_FILE = config['data_paths']['train_file']
    TEST_FILE = config['data_paths']['test_file']
    NEG_RATIO = config['parameters']['neg_ratio']
    SAMPLING = config['parameters']['sampling']

    # Initialize data loader
    pp = PolarsLoader(sampling=SAMPLING)

    # Load and preprocess training data
    train, CAT_COLS, NUM_COLS = pp.initial_preprocessing(
        TRAIN_FILE, is_train=True, neg_ratio=NEG_RATIO
    )

    # Load and preprocess test data
    test, _, _ = pp.initial_preprocessing(
        TEST_FILE, is_train=False
    )

    # Define features and target
    FEATURES = CAT_COLS + NUM_COLS
    TARGET = 'target'

    # Initialize feature engineering
    fe = FeatureEngineering()

    # Process training and test data
    train = fe.proccess(train, CAT_COLS, is_train=True)
    test = fe.proccess(test, CAT_COLS, is_train=False)

    return train, test, FEATURES, TARGET

def train_model_inference(train, test, FEATURES, TARGET, config):
    """Trains the model and returns the trained model and metrics."""
    print("\n","Moving to model training...")
    X = train[FEATURES]
    y = train[TARGET]

    # Model parameters
    lgb_params = config['model_parameters']

    # Training parameters
    training_params = config['training_parameters']
    do_crossvalidation = training_params['do_crossvalidation']
    do_oversampling = training_params['do_oversampling']
    NEG_RATIO = config['parameters']['neg_ratio']
    multiplier = training_params.get('multiplier', 1.5)

    # Calculate UP_RATIO
    UP_RATIO = round((NEG_RATIO / 100) * multiplier, 5)
    if do_oversampling:
        print(f"Oversampling with UP_RATIO: {UP_RATIO}")
    else:
        print(f"No oversampling")

    # Set up the estimator pipeline
    if do_oversampling:
        print(f"Oversampling with UP_RATIO: {UP_RATIO}")
        estimator = Pipeline([
            ('over_sampler', RandomOverSampler(sampling_strategy=UP_RATIO, random_state=training_params['random_state'])),
            ('classifier', lgb.LGBMClassifier(**lgb_params)),
        ])
    else:
        estimator = lgb.LGBMClassifier(**lgb_params)

    metrics = {}

    if do_crossvalidation:
        print("\n","Cross-validating...")

        # Cross-validation parameters
        cv_params = {
            'n_splits': training_params['n_splits'],
            'shuffle': training_params['shuffle'],
            'random_state': training_params['random_state'],
        }
        cv = KFold(**cv_params)

        # Perform cross-validation
        results = cross_validate(
            estimator=estimator,
            X=X, y=y,
            cv=cv,
            scoring=training_params['scoring'],
            return_train_score=True
        )

        # Print and store F1 scores for both train and validation
        print(f"Folds Train F1 scores: {results['train_score']}")
        print(f"Folds Validation F1 scores: {results['test_score']}")
        print(f"Mean Train F1 score: {np.mean(results['train_score'])}")
        print("\n", "*" * 50)
        print(f"Mean Validation F1 score: {np.mean(results['test_score'])}")
        print("*" * 50)

        # Store metrics
        metrics['train_scores'] = results['train_score']
        metrics['valid_scores'] = results['test_score']
        metrics['mean_train_score'] = np.mean(results['train_score'])
        metrics['mean_valid_score'] = np.mean(results['test_score'])
    else:
        print("Skipping cross-validation.")

    # Train the final model on the entire training data
    print("\n", "Training final model...")
    estimator.fit(X, y)

    # Make predictions on the test set
    predictions = estimator.predict(test[FEATURES])

    test_f1 = f1_score(test[TARGET], predictions)    
    print("Final model performance on test data:\n")
    print("\n", "*" * 50)
    print(f"F1 on test data: {round(test_f1, 5)}")
    BENCHMARK = config['parameters']['benchmark_f1']
    improvement = round(100 * ((test_f1 / BENCHMARK) - 1), 2)
    print(f"Improvement over Benchmark: {improvement}%")
    print("*" * 50)

    # Save metrics
    metrics['test_f1_score'] = test_f1
    metrics['improvement_over_benchmark'] = improvement

    return estimator, metrics, predictions

def save_metrics(metrics, predictions, test, model, FEATURES, config):
    """Saves metrics, plots, and predictions to the results folder."""

    # Create results directory if it doesn't exist
    results_folder = config['data_paths']['results_folder']
    os.makedirs(results_folder, exist_ok=True)

    # Save Metrics to a Text File
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(results_folder, f'metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        # Write cross-validation metrics if available
        if 'mean_train_score' in metrics:
            f.write(f"Mean Train F1 Score: {metrics['mean_train_score']:.5f}\n")
        if 'mean_valid_score' in metrics:
            f.write(f"Mean Validation F1 Score: {metrics['mean_valid_score']:.5f}\n")
        # Write test metrics
        if 'test_f1_score' in metrics:
            f.write(f"Test F1 Score: {metrics['test_f1_score']:.5f}\n")
        if 'improvement_over_benchmark' in metrics:
            f.write(f"Improvement over Benchmark: {metrics['improvement_over_benchmark']:.2f}%\n")
    print(f"Metrics saved to {metrics_file}")

    # Save Cross-Validation Plot
    if 'train_scores' in metrics and 'valid_scores' in metrics:
        plot_cross_validation_scores(
            metrics['train_scores'], metrics['valid_scores'], results_folder)

    # Save Feature Importance Plot
    CHECK_FI = config['parameters'].get('check_feature_importance', True)
    if CHECK_FI:
        plot_feature_importances(model, FEATURES, results_folder)

    # # Save Predictions (Optional)
    # test['predictions'] = predictions
    # predictions_file = os.path.join(results_folder, 'test_predictions.csv')
    # test.to_csv(predictions_file, index=False)
    # print(f"Predictions saved to {predictions_file}")

def main():
    # Load configuration
    config = load_config()

    # Process data
    with StringCache():
        train, test, FEATURES, TARGET = process_data(config)
    print("\nData processed successfully.")
    #Print data information
    print("-"*50)
    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print(f"Number of features: {len(FEATURES)}")
    print("-"*50)

    print(f"% target distribution in train data after downsampling:\n {train[TARGET].value_counts(normalize=True)}")
    print("-"*50)

    # Train model and make predictions on test data
    model, metrics, predictions = train_model_inference(train, 
                                                        test, 
                                                        FEATURES, 
                                                        TARGET, 
                                                        config)

    # Save trained model
    model_save_path = config['data_paths']['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Trained model saved at {model_save_path}")

    # Save metrics and predictions
    save_metrics(metrics, predictions, test, model, FEATURES, config)
    
if __name__ == '__main__':
    main()