# Mirbagheri-ML-ProjectWork
This repository provides an implementation for a classification task for machine learning course @unibo.

# Project Description
In this project, I am working with a captivating dataset available on Kaggle, accessible through this [Link](https://www.kaggle.com/competitions/cat-in-the-dat/data). The primary objective of this experiment is to explore various encoding schemes suitable for different categorical features within the dataset. By employing these encoding techniques, I aim to enhance the performance of the classification models. To determine the most promising approach, I apply four distinct machine learning models. To ensure optimal performance, I fine-tune the hyperparameters of each model using GridSearchCV.

## Installation

To run this Jupyter Notebook locally, you'll need to follow these steps:

1. **Clone the repository:** Begin by cloning this repository to your local machine. You can do this by running the following command in your terminal or command prompt:
  git clone https://github.com/shakibaMB/Cat_Classification_Categorical_Feature_Encoding


2. **Requirements:**
   - Python 3.x
   - numpy
   - pandas
   - scikit-learn
   - scipy
   - imbalanced-learn

## Feature Engineering

In this project, I performed various feature engineering techniques to enhance the predictive power of our models. The following feature encoding methods were applied:

- **Manual Encoding:** For certain categorical variables, I manually encoded them by assigning numerical values based on prior knowledge. 

- **One-Hot Encoding:** I utilized one-hot encoding to transform some categorical variables into binary vectors. This technique creates new binary columns for each unique category within a variable, effectively representing the presence or absence of each category as a binary value (0 or 1).

- **Ordinal Encoding:** In some cases, I used ordinal encoding to assign integer values to categorical variables based on their order or rank. This approach is suitable when there is an inherent order or hierarchy among the categories.

These feature encoding techniques were applied to ensure compatibility with machine learning algorithms that require numerical inputs. By appropriately encoding categorical features, we enabled our models to effectively utilize this information for accurate predictions.

## Model Implementation

In this project, I trained and evaluated four different machine learning models on our dataset. The goal was to select the most promising model for final prediction on the test data. The four models used are as follows:

- Decision Tree (DT)
- Logistic Regression
- Random Forest
- k-Nearest Neighbors (KNN)

### Model Tuning

Before fitting the models on the training data, I performed hyperparameter tuning using GridSearchCV on the validation set. This process involved systematically searching for the best combination of hyperparameters that optimize the model's performance.

### Evaluation and Model Selection

Each model was trained on the training set with the tuned hyperparameters, and then evaluated on the validation set. The evaluation metrics used are accuracy, precision, recall, or F1-score.

## Oversampling data

In this project, I encountered an imbalanced dataset, where the distribution of classes was somewhow uneven. To mitigate the challenges posed by this class imbalance, I decided to utilize the SMOTE (Synthetic Minority Over-sampling Technique) oversampling algorithm. By applying SMOTE, I was able to address the issue of limited instances in the minority class by generating synthetic examples that accurately represented the minority class. This approach allowed me to balance the class distribution and provide the model with a more representative training set.
