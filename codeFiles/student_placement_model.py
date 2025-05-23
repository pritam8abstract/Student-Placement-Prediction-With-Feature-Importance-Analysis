import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import xgboost as XGBClassifier
import warnings
warnings.filterwarnings('ignore')

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

print("\nMissing values in test data:")
print(test_features.isnull().sum())

# Basic statistics
print("\nBasic statistics of numerical columns:")
print(train_data.describe())

# Analyzing the target variable distribution
print("\nTarget variable distribution:")
print(train_data['Placement'].value_counts())
print(train_data['Placement'].value_counts(normalize=True) * 100)

# Visualize the target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Placement', data=train_data)
plt.title('Placement Distribution')
plt.xlabel('Placement (0: Placed, 1: Not Placed)')
plt.ylabel('Count')
plt.savefig('placement_distribution.png')
plt.close()

# Correlation analysis for numerical features
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'ID' in numerical_cols:
    numerical_cols.remove('ID')
if 'Placement' in numerical_cols:
    numerical_cols.remove('Placement')

plt.figure(figsize=(12, 10))
correlation_matrix = train_data[numerical_cols + ['Placement']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Handling missing values in Entrance_Test column
# Check unique values in the column
print("\nUnique values in Entrance_Test column:")
print(train_data['Entrance_Test'].value_counts())

# Let's look at Percentile_ET distribution based on Entrance_Test
print("\nPercentile_ET statistics by Entrance_Test:")
print(train_data.groupby('Entrance_Test')['Percentile_ET'].describe())

# Strategy for handling missing values:
# 1. For numerical columns: impute with median
# 2. For categorical columns: impute with most frequent value

# Identify categorical and numerical columns
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
if 'ID' in categorical_cols:
    categorical_cols.remove('ID')

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

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

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split training data into train and validation sets
X = train_data.drop(['ID', 'Placement'], axis=1)
y = train_data['Placement']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models to try
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Dictionary to store model performance
model_scores = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{'-'*50}\nTraining {name}...")
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = pipeline.predict(X_val)
    
    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    model_scores[name] = (accuracy, f1)
    
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()

# Select the best model based on F1 score
best_model_name = max(model_scores, key=lambda x: model_scores[x][1])
best_f1 = model_scores[best_model_name][1]
print(f"\nBest model: {best_model_name} with F1 Score of {best_f1:.4f}")

# Fine-tune the best model
print(f"\n{'-'*50}\nFine-tuning {best_model_name}...")

if best_model_name == 'RandomForest':
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }
elif best_model_name == 'LogisticRegression':
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
        'classifier__solver': ['liblinear', 'saga']
    }
else:  # XGBoost
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }

# Create the base pipeline with the best model
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Perform grid search
grid_search = GridSearchCV(
    best_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model on validation set
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_val)
tuned_f1 = f1_score(y_val, y_pred_tuned)
tuned_accuracy = accuracy_score(y_val, y_pred_tuned)

print(f"\nTuned {best_model_name} - Accuracy: {tuned_accuracy:.4f}, F1 Score: {tuned_f1:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred_tuned))

# Feature importance analysis (if the model supports it)
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
    # Get feature names after preprocessing
    preprocessor.fit(X_train)
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_features)
    
    # Get feature importances
    if best_model_name == 'XGBoost':
        importances = tuned_model.named_steps['classifier'].feature_importances_
    else:
        importances = tuned_model.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],  # Ensure matching length
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 important features:")
    print(importance_df.head(10))

# Final model training on complete training data
print("\nTraining final model on complete training dataset...")
final_model = grid_search.best_estimator_
final_model.fit(X, y)

# Make predictions on test data
print("\nMaking predictions on test data...")
test_predictions = final_model.predict(test_features.drop('ID', axis=1))

# Create submission file
submission = pd.DataFrame({
    'ID': test_features['ID'],
    'Placement': test_predictions
})

submission.to_csv('final_submission.csv', index=False)
print("Submission file created: final_submission.csv")

# Summary of model performance
print("\nSummary of model performance:")
for name, (acc, f1) in model_scores.items():
    print(f"{name:20} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
print(f"\nTuned {best_model_name:20} - Accuracy: {tuned_accuracy:.4f}, F1 Score: {tuned_f1:.4f}")

# Additional insights about important features for placement
print("\nKey insights for placement prediction:")
if 'importance_df' in locals():
    for idx, row in importance_df.head(5).iterrows():
        print(f"- {row['Feature']} is a strong predictor of placement outcome")
