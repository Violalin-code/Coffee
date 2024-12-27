# Coffee Type Recommendation App ☕️
## Overview
This project develops a Coffee Type Recommendation System using a pre-trained machine learning model that predicts the ideal coffee type for users based on various preferences. These preferences include factors such as time of day, coffee strength, sweetness level, milk type, and more. The system leverages Gradio to create an intuitive web interface, allowing users to easily input their preferences and receive real-time coffee recommendations.

## Table of Contents
- [ML Model](#ML-Model)
- [Features](#Features)
- [How to Run the Project](#How-to-Run)
- [Project Files](#Project-Files)
- [Project Workflow](#Project-Workflow)

## ML Model
A Decision Tree Classifier was chosen for its suitability in classification tasks and its low RMSE. The model was trained on tokenized input features and tested for accuracy. The model was selected for its low Root Mean Squared Error (RMSE) score of 1.1597 on the test dataset. The trained model is saved as best_model.pkl and is used to make predictions in the app.

## Features
- Personalized Coffee Recommendations: Predicts the best coffee type based on user preferences like time of day, strength, sweetness, and milk choice.
- Pre-Trained Machine Learning Model: The model is saved as best_model.pkl and is used for real-time predictions.
- Label Encoding: Predictions are mapped back to coffee types using a pre-trained label encoder (label_encoder.pkl).
- Interactive Interface: Gradio powers the user interface, providing an easy-to-use web interface for entering preferences and receiving instant recommendations.

## How to Run
- Install Required Libraries: Ensure that the necessary libraries are installed. The code will automatically install scikit-learn if missing.
- Prepare the Model Files: Place the following files in the same directory as the code:
- best_model.pkl (pre-trained model)
- label_encoder.pkl (label encoder for decoding predictions)
- Run the Code: Execute the script to launch the app.

GUI:https://huggingface.co/spaces/vjl004/CoffeeTake2 

## Project Files
- coffee_recommendation_dataset.xlsx: Dataset containing features (Token_0 to Token_9) and labels for coffee preferences.
- label_encoder.pkl: The saved label encoder for decoding predictions.
- best_model (3).pkl: The best-performing model, selected after training and evaluation.

## Project Workflow
### 1. Data Preprocessing
- Label Encoding: The target variable (Label column) is label-encoded using LabelEncoder to prepare the data for classification.
- One-Hot Encoding: Categorical features (Token_0 to Token_9) are one-hot encoded to create binary variables.
- Data Splitting: The dataset is split into training (80%) and testing (20%) sets.
- Standardization: Features are standardized using StandardScaler for the Support Vector Machine (SVM) model, as SVMs perform better on scaled data.

### 2. Model Training and Evaluation
- Classifiers Used:
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Training: Each classifier is trained on the training data. For SVM, the dataset is standardized, while other models use the original data.
- Evaluation: Model performance is evaluated using Root Mean Squared Error (RMSE) on the test data to assess the closeness of predictions to actual labels.

### 3. Selecting the Best Model
- Best Model Selection: The model with the lowest RMSE score is selected for deployment.
- Saving the Model: The best model is saved as best_model.pkl and used for future predictions. Additionally, label_encoder.pkl is saved for label decoding.
