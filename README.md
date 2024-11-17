# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Dataset**  
   Import and load the dataset for analysis.

2. **Data Preprocessing**  
   Clean and preprocess the data to ensure consistency and accuracy.

3. **Split Data into Training and Testing Sets**  
   Divide the dataset into training and testing subsets for model validation.

4. **Train Logistic Regression Model**  
   Use the training data to fit and train a logistic regression model.

5. **Generate Predictions**  
   Apply the trained model to predict outcomes on the testing data.

6. **Evaluate Model Performance**  
   Assess the model’s accuracy and performance metrics.

7. **Visualize Results**  
   Create visualizations to interpret and present the model's outcomes.

8. **Make Predictions on New Data**  
   Utilize the trained model to make predictions on fresh or unseen data.

## Program:
```

/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Sriram S S
RegisterNumber:  212222230150*/
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
data = pd.read_csv(url)

# Encoding the target variable
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# Selecting features and target
X = data.drop(columns=['class'])
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and Train the Logistic Regression model with increased max_iter
model = LogisticRegression(max_iter=2000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Example prediction (Assuming a new food item with example values)
new_food_item = pd.DataFrame([[120, 2, 0.5, 0.7, 0.1, 0, 15, 150, 20, 5, 10, 3, 100, 30, 40, 5, 0]], 
                             columns=X.columns)  # Ensure columns match training data
new_food_item_scaled = scaler.transform(new_food_item)
pred_class = model.predict(new_food_item_scaled)

print("Predicted Class for New Food Item:", label_encoder.inverse_transform(pred_class)[0])`
```

## Output:
<img width="540" alt="Screenshot 2024-11-14 at 11 13 11 AM" src="https://github.com/user-attachments/assets/d23b1424-c5ac-49e0-adeb-3b0f4426476b">


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
