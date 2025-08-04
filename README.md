# Titanic-Survival-Prediction---Supervised-Machine-Learning-Project\
Objective
Build a machine learning model to:

Analyze features contributing to survival

Train a model using RandomForestClassifier

Predict survival outcome for test data

 What You Learn
Data preprocessing: cleaning, encoding, filling missing values

Model training with scikit-learn

Performance evaluation (accuracy, precision, recall)

Visualizing results with Seaborn and Matplotlib

Evaluate with accuracy and confusion matrix
This project uses supervised learning to predict passenger survival on the Titanic using a real-world dataset. It applies a Random Forest Classifier built with Scikit-learn, based on features like age, sex, class, and fare.
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Data cleaning
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Split data
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

 Project Folder Structure
kotlin
Copy
Edit
Titanic-Survival-ML/
│
├── Titanic-Dataset.csv
├── titanic_model.ipynb
└── README.md  ← (this file)
🧠 Next Steps (Optional Ideas)
Try Logistic Regression or SVM

Perform hyperparameter tuning with GridSearchCV

Visualize decision trees

Export the model using

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
