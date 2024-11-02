<img src="https://img.shields.io/badge/Polars-white?logo=polars" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/pandas-white?logo=pandas&logoColor=250458" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=013243" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/Scikit--learn-white?logo=scikit-learn" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/LightGBM-white?logo=LightGBM" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/MLflow-white?logo=MLflow" style="height: 25px; width: auto;"> <img src="https://img.shields.io/badge/Optuna-white?logo=python" style="height: 25px; width: auto;">

# Ad Exchange Prediction Project

## Project Goal

The aim of this project is to develop a machine learning model to predict user behavior on an ad exchange platform. By accurately forecasting user interactions, the model assists in optimizing ad placements and maximizing revenue.

## Results

The final model achieved a **23% improvement** over the benchmark model in terms of F1 score, indicating a significant enhancement in predictive performance.

## Detailed Report

For a comprehensive project description, methodology, analysis, and insights, please refer to the Jupyter notebook [`approach_and_final_results.ipynb`](approach_and_final_results.ipynb).

## 🚀 Implementation

To set up the environment and run the project:

1. Download the train and test data in the `data/` folder.

2. Create the environment using the provided `environment.yml` file:
      ```bash
      conda env create -f environment.yml
      ```

3. Activate the environment:
      ```bash
      conda activate ad_exchange_pred
      ```

4. Run the main script:
      ```bash
      python main.py
      ```

5. Play around with configuration file parameters.

6. Run mlflow to follow experiments metrics and parameters:
      ```bash
      mlflow ui
      ```