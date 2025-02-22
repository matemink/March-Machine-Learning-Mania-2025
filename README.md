# March Machine Learning Mania 2025: Predictive Modeling with CatBoost 🏀🏆

This repository contains the `march-machine-learning-mania-2025.ipynb` notebook, which implements a CatBoost-based model for predicting NCAA Men's and Women's basketball tournament outcomes for the 2025 competition, hosted on Kaggle.

## Project Overview

This project aims to leverage historical NCAA data and machine learning techniques to forecast the probabilities of all possible tournament matchups. Unlike traditional bracketology, this competition demands a predictive model capable of generalizing across all potential team combinations, rather than just those in the final tournament bracket. Submissions are evaluated using the Brier score, emphasizing the accuracy of predicted probabilities.

The core of this project is to develop a robust predictive model using CatBoost, focusing on:

* **Comprehensive Data Handling:** Loading, cleaning, and merging historical NCAA data from multiple CSV files.
* **Advanced Feature Engineering:** Creating predictive features such as score differentials, seed differences, and aggregated game statistics.
* **CatBoost Model Training:** Utilizing `CatBoostClassifier` with optimized hyperparameters (achieved through Optuna, though the best parameters are currently hardcoded).
* **Tournament Matchup Prediction:** Generating predictions for all possible tournament matchups in the required submission format, evaluated using the Brier score.

## Key Code Functionality

* **Data Loading and Merging:**
    * Uses `glob` and `pandas` to load and combine data from various CSV files (team information, game results, seeds).
    * Merges men's and women's tournament data into unified DataFrames.
* **Feature Engineering:**
    * Calculates `ScoreDiff`, `SeedDiff`, and other relevant features.
    * Aggregates game statistics (e.g., average field goals, rebounds) for each unique matchup.
    * Adds Gender as a feature.
* **Model Training with CatBoost:**
    * Employs `CatBoostClassifier` for prediction.
    * Includes hyperparameter optimization (Optuna, with best parameters currently hardcoded).
    * Handles categorical features effectively.
* **Submission Preparation:**
    * Prepares the submission DataFrame in the format required by the Kaggle competition.

## Dependencies

* `numpy`: Numerical computing.
* `pandas`: Data manipulation and analysis.
* `scikit-learn`: Machine learning tools.
* `catboost`: Gradient boosting framework.
* `optuna`: Hyperparameter optimization.
* `glob`: File path matching.
* `jupyter`: Interactive notebook environment.
