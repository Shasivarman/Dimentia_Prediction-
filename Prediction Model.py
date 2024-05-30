import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('dementia_patients_health_data.csv')

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

# Data preprocessing
def Data_preprocessing(df):
    df['Dosage in mg'] = df['Dosage in mg'].fillna(df['Dosage in mg'].median())
    df['Prescription'] = df['Prescription'].fillna('None')
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

    df1.drop('Diabetic', axis=1, inplace=True)

    scaler = StandardScaler()
    numerical_columns = ["AlcoholLevel", "HeartRate", "BloodOxygenLevel", "BodyTemperature", "Weight", "MRI_Delay",
                         "Dosage in mg", "Age", "Cognitive_Test_Scores"]
    for col in numerical_columns:
        df1[col] = scaler.fit_transform(np.array(df1[col]).reshape(-1, 1))

    print(df1.head(10))
    return df1

df1 = Data_preprocessing(df)


def plot_heatmap(df1):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df1.corr(), annot=True)
    plt.show()

plot_heatmap(df1)

def plot_histograms_with_target(df, features, target):
    # Set up the subplot grid
    num_plots = len(features)
    num_cols = 2
    num_rows = num_plots // num_cols + (1 if num_plots % num_cols > 0 else 0)

    # Set up the figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    # Plot histograms for each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(data=df, x=feature, hue=target, ax=ax, kde=True, multiple="stack", palette="muted")
        ax.set_title(f'Histogram of {feature} by {target}')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Specify features and target variable
features = ['Chronic_Health_Conditions', 'Cognitive_Test_Scores', 'APOE_ε4', 'Depression_Status']
target = 'Dementia'

# Plot histograms for specified features with target variable
plot_histograms_with_target(df1, features, target)

def plot_pairplot(df, features, target):
    # Combine features and target into a single DataFrame
    df_pairplot = df[features + [target]]

    # Create pair plot
    sns.pairplot(df_pairplot, hue=target, palette="husl", diag_kind='kde')
    plt.show()

# Specify features and target variable
features = ['Chronic_Health_Conditions', 'Cognitive_Test_Scores', 'APOE_ε4', 'Depression_Status']
target = 'Dementia'

# Plot pair plot
plot_pairplot(df1, features, target)

def get_user_input():
    print("Please provide the following information:")
    cognitive_test_scores = float(input("Cognitive Test Scores: "))
    apoE_ε4 = int(input("APOE_ε4 (0 for No, 1 for Yes): "))
    Chronic_Health_Conditions = int(input("choose 0- Diabetes/1- Heart Disease/2- Hypertension/3-None: "))
    Depression_Status = int(input("Depression(0 for No, 1 for Yes): "))
    return cognitive_test_scores, apoE_ε4,Chronic_Health_Conditions,Depression_Status

# Get user input
user_input = get_user_input()

# Convert user input tuple into a DataFrame
user_input_df = pd.DataFrame([user_input], columns=['Cognitive_Test_Scores', 'APOE_ε4', 'Chronic_Health_Conditions','Depression_Status'])


def model(df1, user_input_df):
    # Define features and target
    columns_of_interest = ['Cognitive_Test_Scores', 'APOE_ε4', 'Chronic_Health_Conditions','Depression_Status']
    X = df1[columns_of_interest]
    y = df1["Dementia"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost classifier
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = xgb.predict(x_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)

    # Use the model to predict user input
    prediction = xgb.predict(user_input_df)

    return prediction

def evaluate_model(y_true, y_pred):
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

# Use the model to predict
prediction = model(df1, user_input_df)

# Print the prediction result
if prediction[0] == 1:
    print("Based on the provided information, the prediction is that the user is likely to have dementia.")
else:
    print("Based on the provided information, the prediction is that the user is not likely to have dementia.")
