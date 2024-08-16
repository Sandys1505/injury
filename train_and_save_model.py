import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('C:\\Users\\poorn\\Downloads\\Crash_Reporting_-_Drivers_Data.csv')

# Example: Dropping unnecessary columns and handling missing data
df = df.drop(['irrelevant_column1', 'irrelevant_column2'], axis=1)
df.fillna(df.median(), inplace=True)

# Encoding categorical variables if necessary
df['categorical_feature'] = df['categorical_feature'].astype('category').cat.codes

# Splitting features and target variable
X = df.drop('target_column', axis=1)
y = df['target_column']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model as a pickle file
model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
