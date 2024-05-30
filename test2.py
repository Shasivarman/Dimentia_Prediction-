import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_absolute_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('dementia_patients_health_data.csv')

#Data Understanding

# Check for leading whitespaces
leading_whitespace = any(col.startswith(' ') for col in df.columns)

# Check for trailing whitespaces
trailing_whitespace = any(col.endswith(' ') for col in df.columns)

print("Leading Whitespace Detected:", leading_whitespace)
print("Trailing Whitespace Detected:", trailing_whitespace)

# Explicitly print column names
print("Column Names:", df.columns)

# Check for typographical errors in column names
expected_columns = ["Diabetic", "AlcoholLevel", "HeartRate", "BloodOxygenLevel", "BodyTemperature", "Weight",
                    "MRI_Delay", "Prescription", "Dosage in mg", "Age", "Education_Level", "Dominant_Hand",
                    "Gender", "Family_History", "Smoking_Status", "APOE_ε4", "Physical_Activity",
                    "Depression_Status", "Cognitive_Test_Scores", "Medication_History", "Nutrition_Diet",
                    "Sleep_Quality", "Chronic_Health_Conditions", "Dementia"]

# Check for missing columns
missing_columns = [col for col in expected_columns if col not in df.columns]
print("Missing Columns:", missing_columns)

def data_ingestion(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Print the first few rows of the DataFrame
    print("First 10 rows of the DataFrame:")
    print(df.head(10))
    print(df.shape)
    print(df.isna().sum())
    print(df.dtypes)

data_ingestion(df)


def plot_hist_by_dementia(df):
    plt.figure(figsize=(15, len(df.columns) * 10))
    count = 0
    for col in df.drop('Dementia', axis=1).columns:
        count += 1
        plt.subplot(len(df.columns), 1, count)
        sns.histplot(x=col, hue='Dementia', data=df)

    plt.show()



plot_hist_by_dementia(df)

def mean_median_plot(df,column):
  mean = df[column].mean()
  median = df[column].median()
  plt.figure(figsize = (15,10))
  sns.histplot(df[column], kde=True);
  plt.axvline(mean,color='blue', linestyle='--',label="mean")
  plt.axvline(median,color='red',label="median")
  plt.legend()

mean_median_plot(df, "Dosage in mg")

# Data Preparation
def Data_preprocessing(df):
    df['Dosage in mg'] = df['Dosage in mg'].fillna(df['Dosage in mg'].median())
    df['Prescription'] = df['Prescription'].fillna('None')
    df['Chronic_Health_Conditions'] = df['Chronic_Health_Conditions'].fillna('None')
    missing_values = df.isna().sum()
    print(missing_values)

    # Remove leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()

    df1 = df.copy()  # Make a copy of the DataFrame

    encoder = LabelEncoder()
    categorical_columns = ["Prescription", "Education_Level", "Dominant_Hand", "Gender", "Family_History",
                           "Smoking_Status", "APOE_ε4", "Medication_History", "Nutrition_Diet", "Sleep_Quality",
                           "Chronic_Health_Conditions", "Physical_Activity", "Depression_Status"]
    for col in categorical_columns:
        if col in df1.columns:
            df1[col] = encoder.fit_transform(df1[col])

    #df1.drop('Diabetic', axis=1, inplace=True)

    scaler = StandardScaler()
    numerical_columns = ["AlcoholLevel", "HeartRate", "BloodOxygenLevel", "BodyTemperature", "Weight", "MRI_Delay",
                         "Dosage in mg", "Age", "Cognitive_Test_Scores"]
    for col in numerical_columns:
        df1[col] = scaler.fit_transform(np.array(df1[col]).reshape(-1, 1))

    print(df1.head(10))
    return df1

df1 = Data_preprocessing(df)

#Data Modeling
def plot_heatmap(df1):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df1.corr(), annot=True)
    plt.show()

plot_heatmap(df1)
# Specify the target variable
target = 'Dementia'

# Individual Feature Analysis
def plot_histograms_by_dementia(df, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue=target, kde=True, palette='husl', multiple='stack')
        plt.title(f'Histogram of {feature} by {target}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        if len(df[target].unique()) > 1:
            plt.legend(title=target)
        plt.show()

# Correlation Analysis
def plot_correlation_with_dementia(df, features, target):
    correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    plt.title(f'Correlation of Features with {target}')
    plt.xlabel('Correlation')
    plt.ylabel('Feature')
    plt.show()

# Pairwise Relationship
def plot_pairwise_relationship(df, features, target):
    df_pairplot = df[features + [target]]
    if len(df[target].unique()) > 1:
        sns.pairplot(df_pairplot, hue=target, palette='husl', diag_kind='kde')
    else:
        sns.pairplot(df_pairplot, diag_kind='kde')
    plt.show()

# Specify the selected features
selected_features = ['Chronic_Health_Conditions', 'Cognitive_Test_Scores', 'APOE_ε4', 'Depression_Status','Diabetic']

# Perform Individual Feature Analysis
plot_histograms_by_dementia(df1, selected_features)

# Perform Correlation Analysis
plot_correlation_with_dementia(df1, selected_features, target)

# Perform Pairwise Relationship Analysis
plot_pairwise_relationship(df1, selected_features, target)

def get_user_input():
    cognitive_test_scores = int(input("Cognitive Test Scores (must be between 1 and 10): "))
    while cognitive_test_scores < 1 or cognitive_test_scores > 10:
        print("Invalid input. Cognitive Test Scores must be between 1 and 10.")
        cognitive_test_scores = int(input("Cognitive Test Scores (must be between 1 and 10): "))

    apoE_ε4 = int(input("APOE_ε4 (0 for No, 1 for Yes): "))
    while apoE_ε4 not in [0, 1]:
        print("Invalid input. APOE_ε4 must be either 0 or 1.")
        apoE_ε4 = int(input("APOE_ε4 (0 for No, 1 for Yes): "))

    chronic_health_conditions = int(input("Choose Chronic Health Conditions:\n0- Diabetes\n1- Heart Disease\n2- Hypertension\n3- None\nEnter your choice: "))
    while chronic_health_conditions not in [0, 1, 2, 3]:
        print("Invalid input. Choose a number between 0 and 3.")
        chronic_health_conditions = int(input("Choose Chronic Health Conditions:\n0- Diabetes\n1- Heart Disease\n2- Hypertension\n3- None\nEnter your choice: "))

    depression_status = int(input("Depression (0 for No, 1 for Yes): "))
    while depression_status not in [0, 1]:
        print("Invalid input. Depression status must be either 0 or 1.")
        depression_status = int(input("Depression (0 for No, 1 for Yes): "))

    return cognitive_test_scores, apoE_ε4, chronic_health_conditions, depression_status

# Get user input
user_input = get_user_input()

# Convert user input tuple into a DataFrame
user_input_df = pd.DataFrame([user_input], columns=['Cognitive_Test_Scores', 'APOE_ε4', 'Chronic_Health_Conditions', 'Depression_Status'])

#Model Evaluation
def evaluate_model(y_true, y_pred, x, y):
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean Absolute Error:", mae)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(cr)

    # ROC Curve and AUC Score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    print("AUC Score:", auc)

def model(df1, user_input_df):
    columns_of_interest = ['Cognitive_Test_Scores', 'APOE_ε4', 'Chronic_Health_Conditions', 'Depression_Status']
    x = df1[columns_of_interest]
    y = df1["Dementia"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred, x_test, y_test)

    # Use the model to predict 
    prediction = log_reg.predict(user_input_df)

    return prediction


# Use the model to predict
prediction = model(df1, user_input_df)

# Print the prediction result
if prediction[0] == 1:
    print("Based on the provided information, the prediction is that the user is likely to have dementia.")
else:
    print("Based on the provided information, the prediction is that the user is not likely to have dementia.")