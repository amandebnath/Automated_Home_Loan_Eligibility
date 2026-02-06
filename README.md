
# Project Title: Home Loan Prediction System using Machine Learning
## Third Year College Project

<!-- Placeholder for College Logo -->
<img src="https://github.com/amandebnath/Automated_Home_Loan_Eligibility/blob/main/MSIT%20ME%20Logo.png" alt="college department logo" width="200" height="200">

## 1. Executive Summary
This project aims to automate the loan eligibility process for a housing finance company. By analyzing various customer details such as gender, marital status, education, income, and credit history, the system predicts whether a customer is eligible for a loan. This classification task involves data exploration, preprocessing (handling missing values, feature engineering), and the application of several machine learning models to identify the most suitable candidates for loan approval.

## 2. Data Source
The dataset used in this project is sourced from Kaggle. It consists of two main files:
*   `train.csv`: Contains the training data with customer details and their corresponding loan status.
*   `test.csv`: Contains customer details for which loan status needs to be predicted.

These datasets were loaded from a Google Drive path: `/content/drive/MyDrive/Data Science/home loan/`

## 3. College Project Context
This project was developed as part of a B.Tech Third Year coursework, focusing on practical applications of machine learning for real-world business problems. It demonstrates proficiency in data analysis, predictive modeling, and model evaluation techniques taught during the curriculum.

## 4. Tech Stack
The following Python libraries and tools were utilized in this project:
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Seaborn**: For statistical data visualization.
*   **Matplotlib**: For plotting and visualizing data.
*   **Scikit-learn**: For machine learning model implementation (Logistic Regression, Decision Tree, Random Forest) and model selection (GridSearchCV, train_test_split).
*   **XGBoost**: For implementing the XGBoost classification model.

## 5. Project Overview
The project follows a standard machine learning workflow:

### 5.1. Data Loading and Initial Inspection
*   Loading `train.csv` and `test.csv` into pandas DataFrames.
*   Initial checks on data shapes, column names, and data types.

### 5.2. Exploratory Data Analysis (EDA)
*   Analyzing the distribution of the target variable (`Loan_Status`).
*   Exploring categorical and ordinal independent variables using `value_counts()` and bar plots.
*   Visualizing numerical independent variables (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term) using histograms and box plots.
*   Examining the relationship between independent variables and the target variable using cross-tabulations and stacked bar plots.
*   Analyzing correlations between numerical features using a heatmap.

### 5.3. Data Preprocessing
*   **Handling Missing Values**: Imputing missing values in categorical features with their mode and numerical features (LoanAmount) with their median. `Loan_Amount_Term` was imputed with its mode.
*   **Feature Engineering**: 
    *   Creating `TotalIncome` by combining `ApplicantIncome` and `CoapplicantIncome`.
    *   Applying logarithmic transformation to `LoanAmount`, `TotalIncome` to handle skewness.
    *   Calculating `EMI` (Equated Monthly Installment) and `Balance_Income`.
*   **Categorical to Numerical Conversion**: Converting categorical variables into numerical representations using one-hot encoding via `pd.get_dummies()`.
*   **Encoding Target Variable**: Converting `Loan_Status` ('Y'/'N') to numerical (1/0).

## 6. Model Building
The following classification models were implemented and evaluated:

### 6.1. Logistic Regression
*   A baseline model to predict loan approval.

### 6.2. Decision Tree
*   A tree-based model for classification.

### 6.3. Random Forest
*   An ensemble learning method for classification.
*   Hyperparameter tuning was performed using `GridSearchCV` to optimize model performance.

### 6.4. XGBoost
*   A gradient boosting framework known for its efficiency and performance.

## 7. Model Evaluation
Models were evaluated based on their accuracy scores on a validation set (30% of the training data).

*   **Logistic Regression Accuracy**: 78.92%
*   **Decision Tree Accuracy**: 71.35%
*   **Random Forest Accuracy**: 77.84% (tuned with `max_depth=10`, `n_estimators=50`)
*   **Random Forest (GridSearchCV) Accuracy**: 76.76% (best estimator found: `max_depth=7`, `n_estimators=41`)
*   **XGBoost Accuracy**: 77.84%

## 8. Important Features
Feature importances were analyzed using the Random Forest model, providing insights into which variables contribute most significantly to loan approval prediction.

## 9. Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    cd <your-repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn xgboost
    ```
3.  **Place data files:** Ensure `train.csv` and `test.csv` are accessible at the specified path (or modify the code to point to your data location).

## 10. Usage
Execute the Jupyter Notebook cells sequentially to reproduce the analysis, preprocessing, model training, and evaluation steps. The notebook is designed to be self-explanatory, guiding through each stage of the project.

```
