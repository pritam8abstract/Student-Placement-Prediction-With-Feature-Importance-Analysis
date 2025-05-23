# Student-Placement-Prediction-With-Feature-Importance-Analysis

Okay, here's the content converted into a good-looking, README.md compatible format. I've used Markdown for structuring, added code blocks, and tried to maintain the informational flow of the original document.

# Student Placement Prediction: A Machine Learning Approach

**Author:** Dhritabrata Swarnakar
**Email:** pritamswarnakar21@gmail.com
**Date:** May 15, 2025

> A report based on analysis of student academic and skill-related factors for placement prediction.

---

## Abstract

This report presents a comprehensive machine learning approach for predicting student placements based on academic and skill-related factors. The study analyzes a dataset from the Kaggle challenge "Placement Puzzle: Crack the Hiring Code" containing approximately 300 features. We implement and compare four different classification models: Random Forest, Gradient Boosting, Logistic Regression, and XGBoost. Feature importance is examined through correlation analysis, mutual information scores, chi-square tests, and logistic regression coefficients. Results reveal that while the models achieve reasonable overall accuracy (73.33%-80.00%), they struggle with predicting the minority "Not Placed" class, as evidenced by low F1 scores (0.0000-0.2105). The analysis identifies several consistently important predictive features including board examination types, entrance test scores, communication marks, and work experience. The study provides insights into the factors influencing student placement outcomes and suggests future work to address class imbalance and improve predictive performance.

---

## Table of Contents (Summary)

1.  [Introduction](#1-introduction)
2.  [Dataset and Preprocessing](#2-dataset-and-preprocessing)
    *   [Dataset Description](#21-dataset-description)
    *   [Handling Missing Values](#22-handling-missing-values)
    *   [Additional Preprocessing Steps](#23-additional-preprocessing-steps)
3.  [Methodology](#3-methodology)
    *   [Predictive Modeling](#31-predictive-modeling)
    *   [Feature Importance Analysis](#32-feature-importance-analysis)
4.  [Results and Discussion](#4-results-and-discussion)
    *   [Model Performance Analysis](#41-model-performance-analysis)
    *   [Feature Importance Insights](#42-feature-importance-insights)
    *   [Submission Results](#43-submission-results)
5.  [Conclusion and Future Work](#5-conclusion-and-future-work)
    *   [Key Findings](#51-key-findings)
    *   [Limitations](#52-limitations)
    *   [Future Work](#53-future-work)
    *   [Educational Implications](#54-educational-implications)

---

## 1. Introduction

Predicting student placement outcomes is a critical concern in educational institutions, enabling targeted interventions and career guidance for students at risk of not securing placements. This predictive capability allows educational institutions to allocate resources more efficiently and helps students prepare better for their professional journeys.

This study approaches the problem through the lens of the Kaggle challenge "Placement Puzzle: Crack the Hiring Code," which provides a robust dataset for building machine learning models to predict student placement status. The challenge frames the prediction task as a binary classification problem, with placement outcomes coded as 0 (Placed) and 1 (Not Placed).

The primary objective of this report is to:

*   Develop effective machine learning models for placement prediction.
*   Identify key factors influencing placement outcomes.
*   Evaluate model performance with a focus on accurately predicting students at risk of not being placed.
*   Provide insights that can inform educational policies and student support systems.

The F1 score for the minority class (Not Placed) serves as the primary evaluation metric, emphasizing the importance of correctly identifying students who may need additional support for placement success.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset Description

The dataset comprises approximately 300 columns representing various student attributes that can be broadly categorized into:

*   **Academic Performance:** Board examination types (`Board_HSC`, `Board_SSC`), percentage scores across different educational levels (`Percent_SSC`, `Percent_HSC`, `Percent_MBA`).
*   **Entrance Examination:** Entrance test percentiles (`Percentile_ET`), scores (`Entrance_Test`).
*   **Skills Assessment:** Communication skills (`Marks_Communication`), project work evaluation (`Marks_Projectwork`), specific skill tests (`S-TEST`, `S-TEST*SCORE`).
*   **Background Information:** Gender, work experience (`Experience_Yrs`).
*   **Educational Specialization:** MBA specialization (`Specialization_MBA`), HSC stream (`Stream_HSC`).
*   **Target Variable:** Placement status (0 = Placed, 1 = Not Placed).

The dataset structure and preprocessing steps can be better understood by examining the code in `student_placement_model.py`:

```python
# Listing 1: Dataset Loading and Exploration Code
import pandas as pd

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


The code shows that the data is split across multiple CSV files and includes separate training features, training targets, and test features, which is typical of a Kaggle competition format.

2.2 Handling Missing Values

A significant preprocessing challenge in the dataset is the presence of 53 null values in the Entrance_Test column. This feature appears to be particularly important for placement prediction. The student_placement_model.py file reveals the approach used:

# Listing 2: Missing Value Analysis and Handling (Conceptual from student_placement_model.py)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# (Code snippet from document showing parts of a pipeline)
# # Check missing values in Entrance_Test column
# print("\nUnique values in Entrance_Test column:")
# print(train_data['Entrance_Test'].value_counts(dropna=False)) # Added dropna=False for clarity

# # Let's look at Percentile_ET distribution based on Entrance_Test
# print("\nPercentile_ET statistics by Entrance_Test:")
# print(train_data.groupby('Entrance_Test')['Percentile_ET'].describe())

# Strategy for handling missing values:
# 1. For numerical columns: impute with median
# 2. For categorical columns: impute with most frequent value

# Prepare preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Prepare preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


The code shows that a sophisticated approach was taken for handling missing values:

Mean/Median Imputation: Replacing missing values with the central tendency measure.

Mode Imputation: Using the most frequent value, suitable for categorical features.

(Potentially) Model-based Imputation or Creation of Missing Indicator: The document mentions these as possibilities. Based on jupiter_Code.ipynb (not provided here), a statistical imputation was likely employed, potentially with a missing indicator feature.

2.3 Additional Preprocessing Steps

The student_placement_model.py file reveals further preprocessing:

# Listing 3: Feature Preprocessing Pipeline (Conceptual from student_placement_model.py)

# Identify categorical and numerical columns (example logic)
# categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
# if 'ID' in categorical_cols:
#     categorical_cols.remove('ID')
# numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# if 'ID' in numerical_cols:
#     numerical_cols.remove('ID')
# if 'Placement' in numerical_cols: # Assuming 'Placement' is target
#     numerical_cols.remove('Placement')


# Combine preprocessing steps
# (Assuming numerical_transformer and categorical_transformer are defined as in Listing 2)
# And numerical_cols, categorical_cols are defined
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])


The preprocessing pipeline includes:

Automatic Feature Type Detection: Identifying categorical and numerical features.

Categorical Encoding: Using OneHotEncoder with handle_unknown='ignore'.

Feature Scaling: Standardizing numerical features using StandardScaler.

Composite Preprocessing: ColumnTransformer applies steps to different feature types in a unified pipeline.

3. Methodology
3.1 Predictive Modeling

Four distinct machine learning algorithms were implemented:

3.1.1 Random Forest

Selected for robust performance and ability to handle high-dimensional data.

Handles numerical and categorical features.

Provides built-in feature importance.

Reduces overfitting (bagging).

Works well with many features.

3.1.2 Gradient Boosting

Implemented as a sequential ensemble method.

High predictive accuracy.

Flexibility in loss function.

Robust handling of different data types.

Good performance with imbalanced datasets.

3.1.3 Logistic Regression

Selected as a baseline linear model for interpretability and efficiency.

Probabilistic output.

Feature coefficient analysis.

Computationally efficient.

Less prone to overfitting than complex models.

3.1.4 XGBoost

An optimized implementation of gradient boosting.

Regularization to prevent overfitting.

Handles missing values internally.

Efficient parallel processing.

Often superior performance to traditional gradient boosting.

3.2 Feature Importance Analysis

Four techniques were used to identify influential features. (Code snippets from feature_importance_analysis.py)

3.2.1 Correlation Analysis

Measures linear relationships between features and placement outcome.

# Listing 4: Correlation Analysis Code (Conceptual)
# import matplotlib.pyplot as plt
# import seaborn as sns
# Assuming X (features) and y (target) are defined
# plt.figure(figsize=(20, 16))
# correlation_matrix = X.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Feature Correlation Heatmap')
# plt.tight_layout()
# plt.savefig('feature_correlation_heatmap.png')
# plt.close()

# # Print top correlations with Placement
# placement_correlations = X.apply(lambda col: col.corr(y))
# print("Top Correlations with Placement:")
# print(placement_correlations.sort_values(ascending=False).head(10))
# print("\nBottom Correlations with Placement:")
# print(placement_correlations.sort_values(ascending=True).head(10))

3.2.2 Mutual Information Score

Captures linear and non-linear dependencies.

# Listing 5: Mutual Information Analysis Code (Conceptual)
# from sklearn.feature_selection import mutual_info_classif
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def calculate_mutual_information(X, y):
#     mi_scores = mutual_info_classif(X, y)
#     mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information Score': mi_scores})
#     return mi_df.sort_values('Mutual Information Score', ascending=False)

# mi_scores_df = calculate_mutual_information(X, y) # Assuming X, y defined
# print("\nMutual Information Scores:")
# print(mi_scores_df)

# # Visualize Mutual Information Scores
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Mutual Information Score', y='Feature', data=mi_scores_df.head(15))
# plt.title('Top 15 Features by Mutual Information Score')
# plt.tight_layout()
# plt.savefig('mutual_information_scores.png')
# plt.close()

3.2.3 Chi-Square Score

Assesses independence between features (non-negative) and placement outcome.

# Listing 6: Chi-Square Analysis Code (Conceptual)
# from sklearn.feature_selection import chi2, SelectKBest
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Prepare data for chi-square test (non-negative values)
# X_chi2 = X.copy() # Assuming X is defined
# X_chi2 = X_chi2 - X_chi2.min() # Ensure all values are non-negative

# # Perform chi-square test
# chi2_selector = SelectKBest(chi2, k=10) # or k='all'
# chi2_selector.fit(X_chi2, y) # Assuming y is defined

# # Get feature scores
# chi2_scores_df = pd.DataFrame({'Feature': X.columns, 'Chi-Square Score': chi2_selector.scores_})
# chi2_scores_df = chi2_scores_df.sort_values('Chi-Square Score', ascending=False)
# print("\nChi-Square Feature Importance:")
# print(chi2_scores_df)

# # Visualize Chi-Square Scores
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Chi-Square Score', y='Feature', data=chi2_scores_df.head(15))
# plt.title('Top 15 Features by Chi-Square Score')
# plt.tight_layout()
# plt.savefig('chi_square_scores.png')
# plt.close()

3.2.4 Logistic Regression Coefficients

Analyzes feature importance in a linear model context.

# Listing 7: Logistic Regression Coefficient Analysis Code (Conceptual)
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X) # Assuming X is defined

# # Fit Logistic Regression
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_scaled, y) # Assuming y is defined

# # Get feature coefficients
# coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': np.abs(lr.coef_[0])})
# coef_df = coef_df.sort_values('Coefficient', ascending=False)
# print("\nLogistic Regression Feature Coefficients:")
# print(coef_df)

# # Visualize Logistic Regression Coefficients
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15))
# plt.title('Top 15 Features by Logistic Regression Coefficient Magnitude')
# plt.tight_layout()
# plt.savefig('logistic_regression_coefficients.png')
# plt.close()

3.2.5 Comprehensive Feature Analysis

Systematic approach to identify consistently important features across all methods.

# Listing 8: Comprehensive Feature Importance Analysis Code (Conceptual)
# from collections import Counter

# def summarize_feature_importance(): # Assuming various scores_dfs are defined
#     print("\n--- Comprehensive Feature Importance Analysis ---")
#     # Combine different feature importance methods (Example structure)
#     # methods = {
#     #     'Correlation with Placement': placement_correlations,
#     #     'Mutual Information Score': mi_scores_df.set_index('Feature')['Mutual Information Score'],
#     #     'Chi-Square Score': chi2_scores_df.set_index('Feature')['Chi-Square Score'],
#     #     'Logistic Regression Coefficient': coef_df.set_index('Feature')['Coefficient']
#     # }
#     # top_features = {}
#     # for method_name, scores in methods.items():
#     #     top_features[method_name] = list(scores.sort_values(ascending=False).head(10).index)
#     # print("\nTop 10 Features by Different Methods:")
#     # for method, features in top_features.items():
#     #     print(f"\n{method}:")
#     #     for i, feature in enumerate(features, 1):
#     #         print(f"{i}. {feature}")
#     # all_top_features = [feature for features_list in top_features.values() for feature in features_list]
#     # consistent_features = [feat for feat, count in Counter(all_top_features).items() if count > 1] # e.g. appears in >1 list
#     # print("\nConsistently Important Features (appearing in >1 top list):")
#     # for feature in consistent_features:
#     #     print(feature)

# # summarize_feature_importance()

4. Results and Discussion
4.1 Model Performance Analysis
4.1.1 Accuracy and F1 Scores Comparison

Table 1: Model Performance Comparison

Model	Accuracy	F1 Score (Not Placed)
Random Forest	0.8000	0.0000
Gradient Boosting	0.7333	0.1111
Logistic Regression	0.7500	0.2105
XGBoost	0.7667	0.1250

Models achieve reasonable overall accuracy (73.33%-80.00%) but struggle with identifying "Not Placed" students (low F1 scores).
Insights:

Class Imbalance: Dataset likely favors placed students.

Feature Relevance: Current features may not fully capture non-placement factors.

Complexity of Non-Placement: Factors for non-placement might be more diverse.

Code generating these results from student_placement_model.py:

# Listing 9: Model Training and Evaluation Code (Conceptual)
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# # Assuming X, y, preprocessor, and models dictionary are defined
# # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# model_scores = {}
# for name, model_instance in models.items(): # models is a dict like {'Random Forest': RandomForestClassifier(), ...}
#     print(f"\n{'-'*50}\nTraining {name}...")
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor), # Defined in section 2.3
#         ('classifier', model_instance)
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_val)

#     accuracy = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, pos_label=1) # Assuming 1 is "Not Placed"
#     model_scores[name] = (accuracy, f1)

#     print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score (Not Placed): {f1:.4f}")
#     print("Classification Report:")
#     print(classification_report(y_val, y_pred))

#     # Plot confusion matrix
#     cm = confusion_matrix(y_val, y_pred)
#     plt.figure(figsize=(8,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix - {name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.savefig(f'confusion_matrix_{name}.png')
#     plt.close()

Random Forest, despite highest accuracy, completely fails on the "Not Placed" class (F1=0). Logistic Regression, with lower accuracy, has the best F1 score for the "Not Placed" class (0.2105).

4.1.2 Model Behavior Analysis

Conceptual Figure 1: Confusion Matrix for Logistic Regression would show a higher number of true negatives (correctly predicted "Not Placed") compared to other models, despite some false negatives (predicted "Placed" but were "Not Placed").

Models' handling of class imbalance:

Random Forest: Complete bias to majority class.

Gradient Boosting: Modest improvement in minority class prediction.

Logistic Regression: Performs best for minority class, possibly due to linear boundary being less prone to overfitting majority patterns.

XGBoost: Balance between accuracy and minority detection, still biased to majority.

Dataset imbalance addressed during training split using stratify=y:

# Listing 10: Data Preparation and Train-Test Split (Conceptual)
# # Assuming train_data is the merged dataframe from Listing 1
# X = train_data.drop(['ID', 'Placement'], axis=1)
# y = train_data['Placement']

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
#                                                   random_state=42, stratify=y)


stratify=y ensures class distribution is similar in train/validation sets but doesn't fix the fundamental imbalance.

4.2 Feature Importance Insights

Conceptual Figure 2: Top 15 Features by Mutual Information Score would display a bar chart ranking features. Likely candidates from the text include Percent_SSC, Experience_Yrs, Percent_HSC, etc.

4.2.1 Key Predictive Factors

Consistently important features:

Academic Background: Board_HSC, Board_SSC, Stream_HSC, Percent_SSC, Percent_HSC, Percent_MBA.

Assessment Scores: Entrance_Test, S-TEST, S-TEST*SCORE, Percentile_ET.

Skill Indicators: Marks_Communication, Marks_BOCA, Marks_Projectwork.

Professional Experience: Experience_Yrs.

Demographic Factors: Gender.

Educational Specialization: Specialization_MBA.

4.2.2 Interpretation of Important Features

Board Examination Types (Board_HSC, Board_SSC): Reflects varying education quality/recognition.

Entrance Test Performance (Entrance_Test, Percentile_ET): Indicates academic aptitude.

Communication Skills (Marks_Communication): Critical soft skill for employers.

Work Experience (Experience_Yrs): Employers value practical exposure.

MBA Specialization (Specialization_MBA): Reflects market demand.

Gender: Suggests potential gender-based disparities warranting investigation.

Conceptual Figure 3: Feature Correlation Heatmap would show a matrix of correlation coefficients between all pairs of features, highlighting clusters of related academic and skill indicators.

4.3 Submission Results

The final_submission.csv likely contains predicted placement statuses (0 or 1). Based on performance, it might have been generated using Logistic Regression due to its better ability to identify "Not Placed" students, despite slightly lower overall accuracy.

5. Conclusion and Future Work
5.1 Key Findings

Models predict placement with moderate accuracy (73.33%-80.00%) but struggle with "Not Placed" students.

Key predictors: academic background, entrance tests, communication skills, work experience.

Stark discrepancy in overall accuracy vs. F1 for "Not Placed" class highlights challenges for early warning systems.

Logistic Regression outperforms complex models in identifying "Not Placed" students, suggesting complexity isn't always better for minority class detection here.

5.2 Limitations

Missing Value Handling: 53 nulls in Entrance_Test column could introduce bias.

Class Imbalance: Significantly impacts minority class prediction.

Limited Test Set Size: Evaluation metrics may have high variability.

Feature Selection: ~300 columns may include redundant/noisy features.

5.3 Future Work

Advanced Imputation Techniques: Multiple imputation or dedicated Entrance_Test prediction model.

Class Imbalance Mitigation: SMOTE, class weighting, cost-sensitive learning.

Hyperparameter Tuning: Grid search or Bayesian optimization for complex models.

Ensemble Methods: Custom ensembles combining strengths of different models.

Feature Engineering: Interaction terms, transformations based on domain knowledge.

Deep Learning Approaches: Neural networks for complex non-linear relationships.

Temporal Analysis: If multi-cohort data available, incorporate temporal trends.

5.4 Educational Implications

Identified predictors can inform targeted interventions for at-risk students.

Significance of communication skills and project work underscores need for soft skills development alongside technical knowledge.

Continued refinement can contribute to more effective student support systems and improved placement outcomes.

