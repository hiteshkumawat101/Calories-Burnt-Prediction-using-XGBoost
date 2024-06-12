import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

# Load the datasets
exrc_data = pd.read_csv('path_to_your_exercise_data.csv')
cal_data = pd.read_csv('path_to_your_calories_data.csv')

# Merge datasets on 'User_ID'
final_data = pd.merge(cal_data, exrc_data, on='User_ID', how='outer')

# Data preprocessing
final_data.drop('User_ID', axis=1, inplace=True)
final_data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)

# Visualize the data
sns.barplot(x='Gender', y='Calories', data=final_data)
plt.title('Calories Burnt by Gender')
plt.show()

sns.distplot(final_data['Age'])
plt.title('Age Distribution')
plt.show()

correlation = final_data.corr()
sns.heatmap(correlation, fmt='.2f', cmap='Greens', annot=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Define features and labels
X = final_data.drop('Calories', axis=1)
Y = final_data['Calories']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

# Train the XGBoost model
model = XGBRegressor()
model.fit(x_train, y_train)

# Predict and evaluate the model
train_predictions = model.predict(x_train)
train_r2 = metrics.r2_score(y_train, train_predictions)
print("Training R^2 Score:", train_r2)

test_predictions = model.predict(x_test)
test_r2 = metrics.r2_score(y_test, test_predictions)
print("Test R^2 Score:", test_r2)
