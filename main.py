"Confidence Makofane - CodSoft Data Science Internship - Task 3"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
file_path = 'IRIS.csv'
iris_data = pd.read_csv(file_path)

# Inspect the first few rows of the data
print("Dataset Preview:")
print(iris_data.head())

# Encode the species labels if they are categorical
if iris_data['species'].dtype == 'object':
    label_encoder = LabelEncoder()
    iris_data['species'] = label_encoder.fit_transform(iris_data['species'])

# Select features (sepal and petal measurements) and target (species)
X = iris_data.drop(columns=['species'])  # Features
y = iris_data['species']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
