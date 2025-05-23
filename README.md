# Student-Placement-Prediction-With-Feature-Importance-Analysis

**By Dhritabrata Swarnakar**
 pritams******21@gmail.com (*Consider if you want your full email public*)
*May 15, 2025*

> A report based on analysis of student academic and skill-related factors for placement prediction.

**View the full original report: [PDF Version](report/Student_Placement_Prediction_Report.pdf)**

---

## Contents

1.  [Introduction](#1-introduction)
2.  [Dataset and Preprocessing](#2-dataset-and-preprocessing)
    *   [2.1 Dataset Description](#21-dataset-description)
    *   [2.2 Handling Missing Values](#22-handling-missing-values)
    *   [2.3 Additional Preprocessing Steps](#23-additional-preprocessing-steps)
3.  [Methodology](#3-methodology)
    *   [3.1 Predictive Modeling](#31-predictive-modeling)
        *   [3.1.1 Random Forest](#311-random-forest)
        *   [3.1.2 Gradient Boosting](#312-gradient-boosting)
        *   [3.1.3 Logistic Regression](#313-logistic-regression)
        *   [3.1.4 XGBoost](#314-xgboost)
    *   [3.2 Feature Importance Analysis](#32-feature-importance-analysis)
        *   [3.2.1 Correlation Analysis](#321-correlation-analysis)
        *   [3.2.2 Mutual Information Score](#322-mutual-information-score)
        *   [3.2.3 Chi-Square Score](#323-chi-square-score)
        *   [3.2.4 Logistic Regression Coefficients](#324-logistic-regression-coefficients)
        *   [3.2.5 Comprehensive Feature Analysis](#325-comprehensive-feature-analysis)
4.  [Results and Discussion](#4-results-and-discussion)
    *   [4.1 Model Performance Analysis](#41-model-performance-analysis)
        *   [4.1.1 Accuracy and F1 Scores Comparison](#411-accuracy-and-f1-scores-comparison)
        *   [4.1.2 Model Behavior Analysis](#412-model-behavior-analysis)
    *   [4.2 Feature Importance Insights](#42-feature-importance-insights)
        *   [4.2.1 Key Predictive Factors](#421-key-predictive-factors)
        *   [4.2.2 Interpretation of Important Features](#422-interpretation-of-important-features)
    *   [4.3 Submission Results](#43-submission-results)
5.  [Conclusion and Future Work](#5-conclusion-and-future-work)
    *   [5.1 Key Findings](#51-key-findings)
    *   [5.2 Limitations](#52-limitations)
    *   [5.3 Future Work](#53-future-work)
    *   [5.4 Educational Implications](#54-educational-implications)

---

## Abstract

This report presents a comprehensive machine learning approach for predicting student placements based on academic and skill-related factors. The study analyzes a dataset from the Kaggle challenge "Placement Puzzle: Crack the Hiring Code" containing approximately 300 features. We implement and compare four different classification models: Random Forest, Gradient Boosting, Logistic Regression, and XGBoost. Feature importance is examined through correlation analysis, mutual information scores, chi-square tests, and logistic regression coefficients. Results reveal that while the models achieve reasonable overall accuracy (73.33%-80.00%), they struggle with predicting the minority "Not Placed" class, as evidenced by low F1 scores (0.0000-0.2105). The analysis identifies several consistently important predictive features including board examination types, entrance test scores, communication marks, and work experience. The study provides insights into the factors influencing student placement outcomes and suggests future work to address class imbalance and improve predictive performance.

---

## 1. Introduction

Predicting student placement outcomes is a critical concern in educational institutions, enabling targeted interventions and career guidance for students at risk of not securing placements. This predictive capability allows educational institutions to allocate resources more efficiently and helps students prepare better for their professional journeys.

This study approaches the problem through the lens of the Kaggle challenge "Placement Puzzle: Crack the Hiring Code," which provides a robust dataset for building machine learning models to predict student placement status. The challenge frames the prediction task as a binary classification problem, with placement outcomes coded as 0 (Placed) and 1 (Not Placed).

The primary objective of this report is to:
*   Develop effective machine learning models for placement prediction
*   Identify key factors influencing placement outcomes
*   Evaluate model performance with a focus on accurately predicting students at risk of not being placed
*   Provide insights that can inform educational policies and student support systems

The F1 score for the minority class (Not Placed) serves as the primary evaluation metric, emphasizing the importance of correctly identifying students who may need additional support for placement success.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset Description

The dataset comprises approximately 300 columns representing various student attributes that can be broadly categorized into:

*   **Academic Performance:** Board examination types (Board_HSC, Board_SSC), percentage scores across different educational levels (Percent_SSC, Percent_HSC, Percent_MBA)
*   **Entrance Examination:** Entrance test percentiles (Percentile_ET), scores (Entrance_Test)
*   **Skills Assessment:** Communication skills (Marks_Communication), project work evaluation (Marks_Projectwork), specific skill tests (S-TEST, S-TEST\*SCORE)
*   **Background Information:** Gender, work experience (Experience_Yrs)
*   **Educational Specialization:** MBA specialization (Specialization_MBA), HSC stream (Stream_HSC)
*   **Target Variable:** Placement status (0 = Placed, 1 = Not Placed)

The dataset structure and preprocessing steps can be better understood by examining the code in `student_placement_model.py`:

```python
# Listing 1: Dataset Loading and Exploration Code
# Load the data
train_features = pd.read_csv('Train_Features.csv')
train_target = pd.read_csv('Train_Target.csv')
test_features = pd.read_csv('Test_Features.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Display basic information about the datasets
print("Training features shape:", train_features.shape)
print("Training target shape:", train_target.shape)
print("Test features shape:", test_features.shape)
print("Sample submission shape:", sample_submission.shape)

# Merge training features with target
train_data = pd.merge(train_features, train_target, on='ID')

# Check first few rows
print("\nFirst 5 rows of training data:")
print(train_data.head())

# Check data types and missing values
print("\nData types:")
print(train_data.dtypes)

print("\nMissing values in training data:")
print(train_data.isnull().sum())
