# Student-Placement-Prediction-With-Feature-Importance-Analysis
A report based on analysis of student academic and skill-related factors for placement prediction.


Introduction
Predicting student placement outcomes is a critical concern in educational institutions, enabling
targeted interventions and career guidance for students at risk of not securing placements. This
predictive capability allows educational institutions to allocate resources more efficiently and
helps students prepare better for their professional journeys.
This study approaches the problem through the lens of the Kaggle challenge "Placement
Puzzle: Crack the Hiring Code," which provides a robust dataset for building machine learning
models to predict student placement status. The challenge frames the prediction task as a binary
classification problem, with placement outcomes coded as 0 (Placed) and 1 (Not Placed).
The primary objective of this report is to:
• Develop effective machine learning models for placement prediction
• Identify key factors influencing placement outcomes
• Evaluate model performance with a focus on accurately predicting students at risk of not
being placed
• Provide insights that can inform educational policies and student support systems
The F1 score for the minority class (Not Placed) serves as the primary evaluation metric,
emphasizing the importance of correctly identifying students who may need additional support
for placement success.


2 Dataset and Preprocessing
2.1 Dataset Description
The dataset comprises approximately 300 columns representing various student attributes that
can be broadly categorized into:
2
2 Dataset and Preprocessing
• Academic Performance: Board examination types (Board_HSC, Board_SSC), percentage
scores across different educational levels (Percent_SSC, Percent_HSC, Percent_MBA)
• Entrance Examination: Entrance test percentiles (Percentile_ET), scores (Entrance_Test)
• Skills Assessment: Communication skills (Marks_Communication), project work evaluation (Marks_Projectwork), specific skill tests (S-TEST, S-TEST*SCORE)
• Background Information: Gender, work experience (Experience_Yrs)
• Educational Specialization: MBA specialization (Specialization_MBA), HSC stream
(Stream_HSC)
• Target Variable: Placement status (0 = Placed, 1 = Not Placed)
The dataset structure and preprocessing steps can be better understood by examining the
-----------------------------------------------code in student_placement_model.py:-------------------------------------code---------------------------------
# Load the data
2 train_features = pd . read_csv (’ Train_Features .csv ’)
3 train_target = pd . read_csv (’ Train_Target .csv ’)
4 test_features = pd . read_csv (’ Test_Features .csv ’)
5 sample_submission = pd . read_csv (’ sample_submission .csv ’)
6
7 # Display basic information about the datasets
8 print (" Training features shape :", train_features . shape )
9 print (" Training target shape :", train_target . shape )
10 print (" Test features shape :", test_features . shape )
11 print (" Sample submission shape :", sample_submission . shape )
12
13 # Merge training features with target
14 train_data = pd . merge ( train_features , train_target , on =’ID ’)
15
16 # Check first few rows
17 print ("\ nFirst 5 rows of training data :")
18 print ( train_data . head () )
19
20 # Check data types and missing values
21 print ("\ nData types :")
22 print ( train_data . dtypes )
23
24 print ("\ nMissing values in training data :")
25 print ( train_data . isnull () .sum () )
----------------------------------------------------------------------------------------------------------------------------------------
Listing 1: Dataset Loading and Exploration Code
The code shows that the data is split across multiple CSV files and includes separate training
features, training targets, and test features, which is typical of a Kaggle competition format.

2.2 Handling Missing Values
A significant preprocessing challenge in the dataset is the presence of 53 null values in the
Entrance_Test column. This feature appears to be particularly important for placement prediction, as indicated by subsequent feature importance analyses. The student_placement_model.py
file reveals the approach used to handle missing values:
1----------------------------------------------------------------------------------------------------------------code------------------------- 
# Check missing values in Entrance_Test column
2 print ("\ nUnique values in Entrance_Test column :")
3 print ( train_data [’ Entrance_Test ’]. value_counts () )
4
5 # Let ’s look at Percentile_ET distribution based on Entrance_Test
6 print ("\ nPercentile_ET statistics by Entrance_Test :")
7 print ( train_data . groupby (’ Entrance_Test ’) [’ Percentile_ET ’]. describe () )
8
9 # Strategy for handling missing values :
10 # 1. For numerical columns : impute with median
11 # 2. For categorical columns : impute with most frequent value
12
13 # Prepare preprocessing for numerical features
14 numerical_transformer = Pipeline ( steps =[
15 (’imputer ’, SimpleImputer ( strategy =’median ’) ) ,
16 (’scaler ’, StandardScaler () )
17 ])
18
19 # Prepare preprocessing for categorical features
20 categorical_transformer = Pipeline ( steps =[
21 (’imputer ’, SimpleImputer ( strategy =’ most_frequent ’) ) ,
22 (’onehot ’, OneHotEncoder ( handle_unknown =’ignore ’) )])
------------------------------------------------------------------------------------------------------------------------------
Listing 2: Missing Value Analysis and Handling
The code shows that a sophisticated approach was taken for handling missing values:
• Mean/Median Imputation: Replacing missing values with the central tendency measure, which preserves the original distribution but may not capture the relationships between features.
• Mode Imputation: Using the most frequent value, which is suitable for categorical
features but may introduce bias for continuous variables like entrance test scores.
• Model-based Imputation: Predicting missing values using other features, which can
capture complex relationships but risks introducing prediction errors.
• Creation of Missing Indicator: Adding a binary feature indicating whether the entrance test value was missing, which can capture patterns related to non-participation.
Based on the analysis in jupiter_Code.ipynb, a statistical imputation approach was likely
employed, potentially combined with a missing indicator feature to preserve information about
the missing pattern itself.


2.3 Additional Preprocessing Steps
The student_placement_model.py file reveals several additional preprocessing steps that were
implemented:
----------------------------------------------------------------------------------------------------------------------------------Code-----------------------------------
1 # Identify categorical and numerical columns
2 categorical_cols = train_data . select_dtypes ( include =[ ’object ’]) . columns . tolist ()
3 if ’ID ’ in categorical_cols :
4 categorical_cols . remove (’ID ’)
5
6 numerical_cols = train_data . select_dtypes ( include =[ ’int64 ’, ’float64 ’]) . columns .
tolist ()
7 if ’ID ’ in numerical_cols :
8 numerical_cols . remove (’ID ’)
9 if ’Placement ’ in numerical_cols :
10 numerical_cols . remove (’Placement ’)
11
12 # Combine preprocessing steps
13 preprocessor = ColumnTransformer (
14 transformers =[
15 (’num ’, numerical_transformer , numerical_cols ) ,
16 (’cat ’, categorical_transformer , categorical_cols )
17 ])
----------------------------------------------------------------------------------------------------------------------------------------
Listing 3: Feature Preprocessing Pipeline
The preprocessing pipeline includes:
• Automatic Feature Type Detection: The code automatically identifies categorical
and numerical features based on their data types, ensuring appropriate preprocessing for
each.
• Categorical Encoding: Categorical variables are transformed through one-hot encoding
using the OneHotEncoder with handle_unknown=’ignore’ to handle any unseen categories in the test set.
• Feature Scaling: Numerical features are standardized using StandardScaler, which
transforms them to have zero mean and unit variance. This is critical for models like
Logistic Regression that are sensitive to feature scales.
• Composite Preprocessing: The ColumnTransformer applies different preprocessing
steps to different feature types in a unified pipeline, ensuring consistent preprocessing
across training and inference.
The preprocessing pipeline ensures that all features are appropriately transformed before
being fed into the machine learning models, addressing the challenges of mixed data types and
ensuring fair contribution of all features to the prediction task.
