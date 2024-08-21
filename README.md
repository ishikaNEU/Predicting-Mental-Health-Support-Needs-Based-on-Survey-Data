# Predicting-Mental-Health-Support-Needs-Based-on-Survey-Data


# Mental Health Support Prediction

## Overview

This project aims to predict whether an individual may need mental health support based on survey data. The model uses various features such as work environment, mental health history, and demographics to make the predictions. This project is a step towards improving mental health support allocation in workplaces and society.

## Dataset

The dataset used in this project comes from the [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, data cleaning, and model building.
- `src/`: Python scripts for data preprocessing, model training, and evaluation.
- `app/`: A simple Streamlit app for predicting mental health support needs based on user input.

## Model

We used several classification algorithms, including Logistic Regression and Random Forest, to build the model. The final model is selected based on accuracy, precision, recall, and F1-score.

## Usage

To use the Streamlit app locally:

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
