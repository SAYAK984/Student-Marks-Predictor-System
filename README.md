# Student-Marks-Predictor-System

## Project Overview
This project focuses on predicting students' academic performance using Multiple Linear Regression (MLR). Two regression models have been developed and evaluated using a real-world student performance dataset. The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, dimensionality reduction using Principal Component Analysis (PCA), model training, evaluation, and visualization.
The primary objective is to analyze how different academic, demographic, and lifestyle factors influence students' performance and to assess the effectiveness of Linear Regression in predicting academic outcomes.


## Objectives:
1. Predict students' Final Examination Score using academic, demographic, and lifestyle attributes.
2. Predict students' Total Semester Score using assessment-related features.
3. Apply appropriate preprocessing techniques for numerical and categorical data.
4. Perform dimensionality reduction using PCA.
5. Evaluate the regression models using standard performance metrics.
6. Analyze the impact of feature correlation on model performance.

## Dataset:
The dataset contains approximately 5,000 student records with 23 attributes, including:
1. Student demographics
2. Academic performance
3. Attendance
4. Study habits
5. Sleep patterns
6. Stress level
7. Family income
8. Parent education
9. Extracurricular activities
10. Department information


## Data Preprocessing:
The following preprocessing steps were performed:
1. Handling missing values
2. Label Encoding for binary categorical features
3. One-Hot Encoding for multi-class categorical features
4. Train-Test Split
5. Standardization of numerical features using StandardScaler
6. Principal Component Analysis (PCA) on numerical attributes


## Machine Learning Models

# Model-1

**Target Variable**
Final_Score

**Input Features**
1. Academic Scores
2. Attendance
3. Study Hours
4. Sleep Hours
5. Stress Level
6. Age
7. Department
8. Parent Education
9. Family Income
10. Internet Access
11. Extracurricular Activities

# Model-2

**Target Variable**
Total_Score

**Input Features**
1. Midterm Score
2. Final Score
3. Assignment Average
4. Quiz Average
5. Participation Score
6. Project Score
7. Extracurricular Activities



## Model Evaluation Metrics

The models were evaluated using:
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. Coefficient of Determination (R² Score)


## Principal Component Analysis (PCA)

Principal Component Analysis was applied to the numerical features to investigate whether dimensionality reduction could improve model performance.
The cumulative explained variance analysis indicated that the variance was distributed almost uniformly across the numerical features. Consequently, PCA provided only a marginal improvement for Model-1 and was not suitable for Model-2, where the original features directly determine the target variable.



## Key Findings:
* Model-1 achieved a very low R² score, indicating that the available features have limited linear predictive power for estimating the Final Examination Score.
* Correlation analysis showed negligible linear relationships between the predictors and the target variable, explaining the limited performance of Multiple Linear Regression.
* Model-2 successfully predicted the Total Semester Score because the target variable is directly associated with the assessment-related input features.
* PCA was found to be ineffective for significant dimensionality reduction due to the relatively uniform distribution of explained variance across the principal components.



## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn



## Project Workflow

[*Student Dataset*] ---> [*Data Cleaning*] ---> [*Encoding Categorical Features*] ---> [*Train-Test Split*] ---> [*Feature Scaling*] ---> [*Principal Component Analysis*] ---> [*Multiple Linear Regression*] ---> [*Prediction*] ---> [*Performance Evaluation*] ---> [*Visualization & Analysis*]

![Project Workflow](Linear-Regression-Model-Workflow/project_workflow.png)


## Future Improvements:

* Compare Multiple Linear Regression with Decision Tree, Random Forest, and Gradient Boosting Regression.
* Apply feature engineering techniques to improve predictive performance.
* Explore non-linear regression models for datasets exhibiting weak linear relationships.
* Perform cross-validation and hyperparameter optimization.
* Develop an interactive dashboard for visualizing prediction results.
