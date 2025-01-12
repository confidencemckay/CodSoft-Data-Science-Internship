"Confidence Makofane - CodSoft Data Science Internship - Task 2"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
file_path = 'IMDb Movies India.csv'
data = pd.read_csv(file_path , encoding='ISO-8859-1')

# Step 2: Data Cleaning
# Handle missing values
data = data.dropna(subset=['Rating'])  # Drop rows without target variable
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')  # Convert Votes to numeric
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)  # Extract numeric duration
data['Year'] = data['Year'].str.extract('(\d+)').astype(float)  # Extract year
data['Genre'] = data['Genre'].fillna('Unknown')  # Replace missing genres with 'Unknown'

# Fill remaining missing values
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    data[col] = data[col].fillna('Unknown')

# Step 3: Feature Engineering
# One-Hot Encoding for Genre
ohe = OneHotEncoder(sparse_output=False)
genre_encoded = ohe.fit_transform(data[['Genre']])
genre_columns = ohe.get_feature_names_out(['Genre'])
genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)

# Vectorize Director and Actors
vectorizer = TfidfVectorizer(max_features=100)  # Use top 100 terms
director_vec = vectorizer.fit_transform(data['Director']).toarray()
actors_combined = data['Actor 1'] + ' ' + data['Actor 2'] + ' ' + data['Actor 3']
actors_vec = vectorizer.fit_transform(actors_combined).toarray()

# Combine Features
X = pd.concat([data[['Votes', 'Duration', 'Year']], pd.DataFrame(director_vec), pd.DataFrame(actors_vec), genre_df],
              axis=1)
y = data['Rating']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

# Step 7: Plot Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 10 Feature Importances')
plt.show()
