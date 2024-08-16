import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset with dtype and low_memory handling
df = pd.read_csv('sample.csv', low_memory=False)

# Drop unnecessary columns
df1 = df.drop(['Route Type', 'Road Name', 'Report Number', 'Vehicle First Impact Location', 
               'Vehicle Second Impact Location', 'Crash Date/Time', 'Local Case Number', 
               'Agency Name', 'ACRS Report Type', 'Municipality', 'Latitude', 'Longitude', 
               'Location', 'Driverless Vehicle', 'Parked Vehicle', 'Vehicle Year', 
               'Vehicle Make', 'Vehicle Model', 'Equipment Problems', 
               'Non-Motorist Substance Abuse', 'Person ID', 'Circumstance', 'Vehicle ID', 
               'Vehicle Body Type', 'Drivers License State', 'Cross-Street Name', 
               'Off-Road Description', 'Related Non-Motorist'], axis=1)

# Define important columns
important_columns = ['Cross-Street Type', 'Collision Type', 'Weather', 'Surface Condition', 
                     'Light', 'Traffic Control', 'Driver Substance Abuse', 'Driver At Fault', 
                     'Injury Severity', 'Driver Distracted By', 'Vehicle Damage Extent', 
                     'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Speed Limit']

# Create a DataFrame with important columns
df2 = df1[important_columns].copy()

# Fill null values in important columns
for col in df2.columns:
    if df2[col].dtype in [np.float64, np.int64]:  # Numeric columns
        df2[col] = df2[col].fillna(df2[col].median())
    else:  # Non-numeric columns
        df2[col] = df2[col].fillna(df2[col].mode()[0])

# Encode categorical columns
label_encoder = LabelEncoder()
for col in df2.select_dtypes(include='object').columns:
    df2[col] = label_encoder.fit_transform(df2[col])

# Split data into features and target variable
X = df2.drop('Injury Severity', axis=1)
y = df2['Injury Severity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Initialize and train Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)
predictions_lr = logistic_regression_model.predict(X_test)

# Initialize and train Decision Tree Classifier model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
predictions_dc = dt_model.predict(X_test)

# Initialize and train Random Forest Classifier model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
predictions_rf = random_forest_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Evaluation:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model(y_test, predictions_lr, "Logistic Regression")
evaluate_model(y_test, predictions_dc, "Decision Tree Classifier")
evaluate_model(y_test, predictions_rf, "Random Forest")

# Sample new data for prediction
new_data = {
    'Cross-Street Type': [7],
    'Weather': [11],
    'Surface Condition': [4],
    'Traffic Control': [6],
    'Vehicle Movement': [17],
    'Speed Limit': [8]
}

# Create a DataFrame for new data
df_new = pd.DataFrame(new_data)

# Use the trained Logistic Regression model for prediction
new_predictions = logistic_regression_model.predict(df_new)
injury_labels = {
    3: "FATAL INJURY",
    1: "NO APPARENT INJURY",
    2: "SUSPECTED MINOR INJURY",
    4: "POSSIBLE INJURY",
    0: "SUSPECTED SERIOUS INJURY"
}

# Display prediction
for prediction in new_predictions:
    print(f"Prediction: {injury_labels.get(prediction, 'UNKNOWN')}")

# Save the trained model
model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(logistic_regression_model, f)

print("Model trained and saved as model.pkl")
