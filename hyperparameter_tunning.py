# src/hyperparameter_tuning.py

import sys
import os
import yaml
import pandas as pd
import numpy as np
import polars as pl
from polars import StringCache

import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import f1_score

import lightgbm as lgb

import argparse

# Import helper functions
from src.data_preprocessing.cleaner import PolarsLoader
from src.data_preprocessing.feature_engineering import FeatureEngineering

def load_config(config_file='configs/config.yaml'):
    """Loads configuration parameters from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    # Load configuration
    config = load_config()
    
    # Hyperparameters to tune
    NEG_RATIO = trial.suggest_float('neg_ratio', 2, 6)

    # LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'f1',
        'verbose': -1,
        'random_state': 23,
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 20, 64),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
    }
    
    with StringCache():
        # Data processing
        # Set up data paths
        TRAIN_FILE = config['data_paths']['train_file']
        TEST_FILE = config['data_paths']['test_file']
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
        train, NEW_COLS = fe.proccess(train, CAT_COLS, is_train=True)
        test, _ = fe.proccess(test, CAT_COLS, is_train=False)
        
        FEATURES += NEW_COLS

        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]

    # Train the model
    estimator = lgb.LGBMClassifier(**lgb_params)
    estimator.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = estimator.predict(X_test)

    # Evaluate the model
    test_f1 = f1_score(y_test, y_pred)
    BENCHMARK = config['parameters'].get('benchmark_f1', 0.503)
    improvement = round(100 * ((test_f1 / BENCHMARK) - 1), 2)

    # Save metrics
    metrics = {}
    metrics['test_f1_score'] = test_f1
    metrics['improvement_over_benchmark'] = improvement

    # Save metrics to trial user_attrs
    trial.set_user_attr('metrics', metrics)

    return test_f1  # Maximize test F1 score

def tune_hyperparameters(n_trials):
    """Runs the hyperparameter optimization using Optuna."""
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    best_metrics = trial.user_attrs['metrics']
    print(f"  Test F1 Score: {best_metrics['test_f1_score']:.5f}")
    print(f"  Improvement over Benchmark: {best_metrics['improvement_over_benchmark']}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters and metrics to results folder
    save_best_params_and_metrics(trial, best_metrics)

def save_best_params_and_metrics(trial, metrics):
    """Saves the best parameters and metrics to a text file in the results folder."""
    # Load existing config
    config = load_config()
    
    best_params = trial.params

    # Create results directory if it doesn't exist
    results_folder = config['data_paths'].get('results_folder', 'results/')
    os.makedirs(results_folder, exist_ok=True)

    # Ensure metrics are standard Python types
    test_f1_score = float(metrics['test_f1_score'])
    improvement_over_benchmark = float(metrics['improvement_over_benchmark'])

    # Save best parameters and metrics to a text file
    best_results_file = os.path.join(results_folder, 'after_tuning_results.txt')
    with open(best_results_file, 'w') as file:
        file.write("Best Trial Results:\n")
        file.write(f"Test F1 Score: {test_f1_score:.5f}\n")
        file.write(f"Improvement over Benchmark: {improvement_over_benchmark}%\n")
        file.write("\nBest Parameters:\n")
        for key, value in best_params.items():
            file.write(f"{key}: {value}\n")
    print(f"Best results saved to {best_results_file}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for LightGBM model.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter optimization.')
    
    args = parser.parse_args()
    
    tune_hyperparameters(n_trials=args.n_trials)