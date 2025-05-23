import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Load the data
train_features = pd.read_csv('Train_Features.csv')
train_target = pd.read_csv('Train_Target.csv')

# Merge features and target
train_data = pd.merge(train_features, train_target, on='ID')

# Preprocessing for correlation and feature importance analysis
def preprocess_data(df):
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Handle categorical variables
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        # Fill missing values with most frequent value
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        # Encode categorical variables
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
    
    # Fill missing numerical values with median
    numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    return processed_df

# Preprocess the data
processed_data = preprocess_data(train_data)

# Separate features and target
X = processed_data.drop(['ID', 'Placement'], axis=1)
y = processed_data['Placement']

# 1. Correlation Analysis
plt.figure(figsize=(20, 16))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
plt.close()

# Print top correlations with Placement
placement_correlations = X.apply(lambda col: col.corr(y))
print("Top Correlations with Placement:")
print(placement_correlations.sort_values(ascending=False).head(10))
print("\nBottom Correlations with Placement:")
print(placement_correlations.sort_values(ascending=True).head(10))

# 2. Mutual Information Analysis
def calculate_mutual_information(X, y):
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Create a dataframe of features and their mutual information scores
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information Score': mi_scores
    })
    
    # Sort by mutual information score in descending order
    return mi_df.sort_values('Mutual Information Score', ascending=False)

# Calculate mutual information
mi_scores = calculate_mutual_information(X, y)
print("\nMutual Information Scores:")
print(mi_scores)

# Visualize Mutual Information Scores
plt.figure(figsize=(12, 8))
sns.barplot(x='Mutual Information Score', y='Feature', data=mi_scores.head(15))
plt.title('Top 15 Features by Mutual Information Score')
plt.tight_layout()
plt.savefig('mutual_information_scores.png')
plt.close()

# 3. Chi-Square Feature Selection (for categorical variables)
from sklearn.feature_selection import chi2, SelectKBest

# Prepare data for chi-square test (non-negative values)
X_chi2 = X.copy()
X_chi2 = X_chi2 - X_chi2.min()

# Perform chi-square test
chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(X_chi2, y)

# Get feature scores
chi2_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi-Square Score': chi2_selector.scores_
})
chi2_scores = chi2_scores.sort_values('Chi-Square Score', ascending=False)

print("\nChi-Square Feature Importance:")
print(chi2_scores)

# Visualize Chi-Square Scores
plt.figure(figsize=(12, 8))
sns.barplot(x='Chi-Square Score', y='Feature', data=chi2_scores.head(15))
plt.title('Top 15 Features by Chi-Square Score')
plt.tight_layout()
plt.savefig('chi_square_scores.png')
plt.close()

# 4. Logistic Regression Coefficients
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)

# Get feature coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(lr.coef_[0])
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

print("\nLogistic Regression Feature Coefficients:")
print(coef_df)

# Visualize Logistic Regression Coefficients
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15))
plt.title('Top 15 Features by Logistic Regression Coefficient Magnitude')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png')
plt.close()

# Comprehensive Summary
def summarize_feature_importance():
    print("\n--- Comprehensive Feature Importance Analysis ---")
    
    # Combine different feature importance methods
    methods = {
        'Correlation with Placement': placement_correlations,
        'Mutual Information Score': mi_scores.set_index('Feature')['Mutual Information Score'],
        'Chi-Square Score': chi2_scores.set_index('Feature')['Chi-Square Score'],
        'Logistic Regression Coefficient': coef_df.set_index('Feature')['Coefficient']
    }
    
    # Find common top features across methods
    top_features = {}
    for method_name, scores in methods.items():
        top_features[method_name] = list(scores.sort_values(ascending=False).head(10).index)
    
    print("\nTop 10 Features by Different Methods:")
    for method, features in top_features.items():
        print(f"\n{method}:")
        for i, feature in enumerate(features, 1):
            print(f"{i}. {feature}")
    
    # Find features that appear consistently across methods
    from collections import Counter
    all_top_features = [feature for features in top_features.values() for feature in features]
    consistent_features = [feat for feat, count in Counter(all_top_features).items() if count > 1]
    
    print("\nConsistently Important Features:")
    for feature in consistent_features:
        print(feature)

# Run summary
summarize_feature_importance()
