
# Hotel Booking Cancellation Prediction

End-to-end machine learning project for predicting hotel booking cancellations with a Streamlit prediction application.

A machine learning classification project that predicts whether a hotel booking is likely to be cancelled based on booking-related information.

This project covers the complete machine learning workflow from data ingestion and exploratory data analysis to model training, evaluation and prediction. I also built a Streamlit application to use the trained model for making predictions.

## Project Objective

Hotel booking cancellations can affect hotel operations, planning and revenue.

The objective of this project is to build a machine learning model that can predict whether a booking will be cancelled based on information such as lead time, number of guests, market segment, deposit type, previous cancellations, booking changes and other booking details.

## Project Workflow

The project follows these stages:

1. Data Ingestion
2. Exploratory Data Analysis (EDA)
3. Data Validation
4. Data Transformation
5. Model Training
6. Model Evaluation
7. Prediction Pipeline
8. Streamlit Application

## Machine Learning Models

I trained and compared the following classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

## Model Results

The models were compared using Accuracy, Precision, Recall, Specificity and F1 Score.

Model Comparison:
                 Model  Accuracy  Precision  Recall  Specificity  F1 Score  
0  Logistic Regression    0.8277     0.7416  0.8210       0.8316    0.7793   
1        Decision Tree    0.8680     0.8162  0.8306       0.8899    0.8233   
2        Random Forest    0.8923     0.8698  0.8341       0.9266    0.8516   
3                  SVM    0.8268     0.7393  0.8225       0.8293    0.7787   
4                  KNN    0.8602     0.8213  0.7957       0.8982    0.8083   
5          Naive Bayes    0.6931     0.5574  0.8323       0.6112    0.6677   


### Best Performing Model

Among the models tested, **Random Forest** performed best based on the evaluation results.

Random Forest achieved:

- Accuracy: **89.23%**
- Precision: **86.98%**
- Recall: **83.41%**
- Specificity: **92.66%**
- F1 Score: **85.16%**
- ROC-AUC: **0.9596**
- PR-AUC: **0.9409**
- Log Loss: **0.2798**

The trained Random Forest model was saved and used in the prediction pipeline.

## Streamlit Application

I also built a Streamlit application that allows users to enter hotel booking details and get a cancellation prediction.

The application includes inputs such as:

- Hotel type
- Lead time
- Arrival year
- Arrival month
- Arrival week number
- Arrival day
- Weekend nights
- Week nights
- Number of adults
- Number of children
- Number of babies
- Meal
- Country
- Market segment
- Distribution channel
- Reserved room type
- Assigned room type
- Deposit type
- Customer type
- Previous cancellations
- Previous bookings not cancelled
- Booking changes
- Days in waiting list
- Required car parking spaces
- Total special requests
- ADR
- Agent
- Company
- Repeated guest

After entering the booking information, the application displays:

- Cancellation prediction
- Cancellation probability

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- PyYAML
- Joblib
- Matplotlib
- Seaborn


