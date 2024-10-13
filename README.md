README: Coffee Type Recommendation App

OVERVIEW

This project implements a Coffee Type Recommendation System using a pre-trained machine learning model. 
The model predicts the ideal coffee type for a user based on inputs such as time of day, coffee strength, sweetness level, milk type, and more. 
The system uses Gradio to provide a user-friendly web interface where users can input their preferences and get a recommendation.

FEATURES

Predicts the ideal coffee type based on user preferences.
Uses a pre-trained machine learning model stored in a pickle file (best_model (3).pkl).
Utilizes label encoding to map predictions back to coffee types.
Gradio interface allows users to easily input their preferences via dropdown menus and receive recommendations in real-time.

HOW TO RUN

1. Ensure that the required libraries are installed. The code automatically installs scikit-learn if missing.

2. Place the pre-trained model (best_model (3).pkl) and label encoder (label_encoder (3).pkl) in the same directory as the code.

3. Run the code
