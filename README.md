Clinical Trials Analysis

Introduction

This project focuses on analyzing and predicting the outcomes of clinical trials based on a dataset of clinical studies. The goal is to assist researchers in identifying potential challenges in completing trials and improving resource allocation. Using advanced data science techniques, this project extracts meaningful insights and predicts whether a trial will be completed successfully.

The pipeline leverages advanced techniques such as SMOTE for handling class imbalance, hyperparameter tuning, PCA for dimensionality reduction, and ensemble methods for improved model performance.

Key Features

Data Cleaning: Handles missing values, normalizes date formats, and computes derived features such as duration of studies.

Feature Engineering: Adds new features, scales numerical features, and encodes categorical data.

Model Training and Evaluation: Trains multiple classifiers (e.g., Random Forest, Gradient Boosting, Logistic Regression, Voting Classifier) with hyperparameter tuning and evaluates performance using metrics like accuracy, precision, and recall.

Explainability: Provides SHAP visualizations to interpret model predictions.

Reproducibility: Includes random seed settings for consistent results.

Example Outputs: 
Preprocessed Data: preprocessed_clinical_trials.csv
Performance Metrics:
Random Forest: Accuracy 85%, Precision 82%, Recall 87%
Gradient Boosting: Accuracy 88%, Precision 85%, Recall 90%
Visualization Plots:
Correlation heatmaps
SHAP visualizations
Predicted vs Actual charts

Setup Instructions

1. Clone the Repository

git clone <repository-url>
cd NEST

2. Install Dependencies

Using requirements.txt:

pip install -r requirements.txt

Using environment.yml (Conda):

conda env create -f environment.yml
conda activate NEST

3. Run the Code

Upload the dataset (Dataset.xlsx) to the working directory.

Execute the main script in a Jupyter Notebook or Python environment:

python finalNEST.ipynb

4. Outputs

Preprocessed Data: preprocessed_clinical_trials.csv

Visualization Plots: Correlation heatmaps, confusion matrices, SHAP visualizations.

Performance Metrics: Displayed for each model in the terminal.

Reproducibility

Random seeds are explicitly set to ensure reproducible results across different runs. Below are the configurations:

# Random Seed Settings
import numpy as np
import random
import tensorflow as tf

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

File Structure

project-folder/
|-- Dataset.xlsx                  # Input dataset
|-- finalNEST.ipynb               # Main code for analysis with all the plots included in it. This file is #downloaded form google colab which is done by me.
|-- requirements.txt              # Required Python libraries
|-- environment.yml               # Conda environment configuration
|-- README.md                     # Documentation
|-- result/
    |-- preprocessed_clinical_trials.csv # Preprocessed data

Contact

For any queries, please contact:

Name: Shakshi Yadav

Email: shakshi0803@gmail.com