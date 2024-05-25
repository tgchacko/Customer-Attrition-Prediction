# Customer Churn Prediction for a Telecommunications Company

## Table of Contents

[Project Overview](#project-overview)

[Data Sources](#data-sources)

[Data Description](#data-description)

[Tools](#tools)

[EDA Steps](#eda-steps)

[Data Preprocessing Steps and Inspiration](#data-preprocessing-steps-and-inspiration)

[Graphs/Visualizations](#graphs-visualizations)

[Choosing the Algorithm for the Project](#choosing-the-algorithm-for-the-best-project)

[Assumptions](#assumptions)

[Model Evaluation Metrics](#model-evaluation-metrics)

[Results](#results)

[Recommendations](#recommendations)

[Limitations](#limitations)

[Future Possibilities of the Project](#future-possibilities-of-the-project)

[References](#references)

## Project Overview

This project aims to predict customer attrition for a telecommunications company using various machine learning algorithms. The focus is on developing models that can accurately identify customers who are likely to leave based on their historical data.

## Data Sources

The dataset used in this project contains customer information, including demographic details, service usage patterns, and historical behavior. The data is sourced from a CSV file. The dataset has 7043 rows and 21 columns.

[Dataset](https://github.com/tgchacko/Customer-Attrition-Prediction/blob/main/customer_churn.csv)

## Data Description

- **customerID**: Unique identifier for each customer.
- **gender**: Gender of the customer.
- **SeniorCitizen**: Indicates if the customer is a senior citizen (1) or not (0).
- **Partner**: Indicates if the customer has a partner (Yes/No).
- **Dependents**: Indicates if the customer has dependents (Yes/No).
- **tenure**: Number of months the customer has stayed with the company.
- **PhoneService**: Indicates if the customer has phone service (Yes/No).
- **MultipleLines**: Indicates if the customer has multiple lines (Yes/No, No phone service).
- **InternetService**: Type of internet service (DSL, Fiber optic, No).
- **OnlineSecurity**: Indicates if the customer has online security (Yes/No, No internet service).
- **OnlineBackup**: Indicates if the customer has online backup (Yes/No, No internet service).
- **DeviceProtection**: Indicates if the customer has device protection (Yes/No, No internet service).
- **TechSupport**: Indicates if the customer has tech support (Yes/No, No internet service).
- **StreamingTV**: Indicates if the customer has streaming TV (Yes/No, No internet service).
- **StreamingMovies**: Indicates if the customer has streaming movies (Yes/No, No internet service).
- **Contract**: Type of contract (Month-to-month, One year, Two year).
- **PaperlessBilling**: Indicates if the customer has paperless billing (Yes/No).
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- **MonthlyCharges**: The amount charged to the customer monthly.
- **TotalCharges**: The total amount charged to the customer.
- **Churn**: Indicates if the customer has churned (Yes/No).

  ![Dataset Description1](https://i.postimg.cc/GpVrj80K/Screenshot-2024-05-25-at-20-05-18-churn-final-Jupyter-Notebook.png)

  ![Dataset Description2](https://i.postimg.cc/L4LSS8fY/Screenshot-2024-05-25-at-20-08-30-churn-final-Jupyter-Notebook.png)

## Tools

- Python: Data Cleaning and Analysis

    [Download Python](https://www.python.org/downloads/)

- Jupyter Notebook: For interactive data analysis and visualization

    [Install Jupyter](https://jupyter.org/install)
          
**Libraries**

Below are the links for details and commands (if required) to install the necessary Python packages:
- **pandas**: Go to [Pandas Installation](https://pypi.org/project/pandas/) or use command: `pip install pandas`
- **numpy**: Go to [NumPy Installation](https://pypi.org/project/numpy/) or use command: `pip install numpy`
- **matplotlib**: Go to [Matplotlib Installation](https://pypi.org/project/matplotlib/) or use command: `pip install matplotlib`
- **seaborn**: Go to [Seaborn Installation](https://pypi.org/project/seaborn/) or use command: `pip install seaborn`
- **scikit-learn**: Go to [Scikit-Learn Installation](https://pypi.org/project/scikit-learn/) or use command: `pip install scikit-learn`
- **XGBoost**: Go to [XGBoost Installation](https://pypi.org/project/xgboost/) or use command: pip install xgboost
-	**Imbalanced-learn**: Go to [Imbalanced-learn Installation](https://pypi.org/project/imbalanced-learn/) or use command: pip install imbalanced-learn

## EDA Steps

- Data loading and initial exploration
- Data cleaning and manipulation
- Data visualization to understand feature distributions and relationships
- Identifying and handling missing values
- Checking for data imbalances

## Data Preprocessing Steps and Inspiration

- **Handling Missing Values**: Missing values in the 'TotalCharges' column were converted to numeric and missing entries were removed.
- **Encoding Categorical Variables**: Label encoding was applied to convert categorical variables into numeric format.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: It is used to address class imbalance by generating synthetic samples for the minority class, thus balancing the class distribution
- **Scaling Numerical Features**: StandardScaler was used to standardize numerical features.
- **Splitting the Dataset**: The dataset was split into training and testing sets using train_test_split.
- **Feature Selection Techniques**:
o	**PCA (Principal Component Analysis)**: Reduces dimensionality by transforming features into a set of linearly uncorrelated components.
o	**LDA (Linear Discriminant Analysis)**: Maximizes separability between classes by projecting data in a lower-dimensional space.
o	**RFE (Recursive Feature Elimination)**: Selects features by recursively considering smaller sets and pruning the least important ones.
o	**VIF (Variance Inflation Factor)**: Identifies multicollinearity among features. Features with high VIF values were removed to improve model performance.
o	**Feature Importance**: Feature importance methods help identify which features in a dataset contribute the most to the predictive power of a machine learning model.


## Graphs/Visualizations

- **Histograms and bar charts for feature distributions**
  
  ![Distribution1](https://i.postimg.cc/vZ6psQc2/Screenshot-2024-05-25-at-20-10-18-churn-final-Jupyter-Notebook.png)

  ![Distribution2](https://i.postimg.cc/nhPSMNQj/Screenshot-2024-05-25-at-20-12-13-churn-final-Jupyter-Notebook.png)

  ![Distribution3](https://i.postimg.cc/cJcmwjvB/Screenshot-2024-05-25-at-20-13-54-churn-final-Jupyter-Notebook.png)

  ![Distribution4](https://i.postimg.cc/Dyv0CMpM/Screenshot-2024-05-25-at-20-18-09-churn-final-Jupyter-Notebook.png)

  ![Distribution5](https://i.postimg.cc/QtjCRYxC/Screenshot-2024-05-25-at-20-19-07-churn-final-Jupyter-Notebook.png)

  ![Distribution6](https://i.postimg.cc/CK2Y4vB9/Screenshot-2024-05-25-at-20-20-18-churn-final-Jupyter-Notebook.png)

  ![Distribution7](https://i.postimg.cc/xdvB4nCr/Screenshot-2024-05-25-at-20-22-46-churn-final-Jupyter-Notebook.png)
  
- **Correlation heatmap to identify relationships between features**

  ![Heatmap](https://i.postimg.cc/m2QLsfhw/Screenshot-2024-05-25-at-20-24-20-churn-final-Jupyter-Notebook.png)
  
- **Box Plots and Scatter Plots to visualize relationships between multiple features**

  ![Tenure by Internet Service](https://i.postimg.cc/qMbzYHF5/Screenshot-2024-05-25-at-20-27-35-churn-final-Jupyter-Notebook.png)

  !['Tenure vs Monthly Charges'](https://i.postimg.cc/nz3XJ8zr/Screenshot-2024-05-25-at-20-29-20-churn-final-Jupyter-Notebook.png)

  ![Tenure by Contract](https://i.postimg.cc/rpShgjpQ/Screenshot-2024-05-25-at-20-31-01-churn-final-Jupyter-Notebook.png)

  ![Total Charges vs Monthly Charges](https://i.postimg.cc/28GLdZmh/Screenshot-2024-05-25-at-20-32-39-churn-final-Jupyter-Notebook.png)

  ![Monthly Charges by Payment Method](https://i.postimg.cc/tJ0vzc3S/Screenshot-2024-05-25-at-20-34-58-churn-final-Jupyter-Notebook.png)
  
- **Box plots to identify outliers**

  ![Boxplot1](https://i.postimg.cc/wMGvPhBG/Screenshot-2024-05-25-at-20-36-04-churn-final-Jupyter-Notebook.png)

  ![Boxplot2](https://i.postimg.cc/hPD0TmNR/Screenshot-2024-05-25-at-20-52-44-churn-final-Jupyter-Notebook.png)

## Choosing the Algorithm for the Project

- **Logistic Regression**: A statistical model that estimates the probability of a binary outcome based on one or more predictor variables.
- **Decision Tree Classifier**: A tree-like model used to make decisions based on the features of the data.
- **Random Forest Classifier**: An ensemble method that creates a forest of decision trees and outputs the mode of their predictions.
- **K-Neighbors Classifier (KNN)**: A simple, instance-based learning algorithm that assigns class labels based on the majority vote of its neighbors.
- **Naive Bayes Classifier**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions.
- **XGBoost Classifier**: An optimized gradient boosting library designed to be highly efficient and flexible.

## Assumptions

- The dataset provided is representative of the customer base.
- All relevant features are included in the dataset.
- The preprocessing steps adequately prepare the data for modeling.

## Model Evaluation Metrics

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **ROC-AUC Score**: A performance measurement for classification problems at various threshold settings.

## Results

After analyzing the metrics for all the models and feature extraction methods, the overall best model was determined to be the Random Forest Classifier (after hypertuning) without any feature extraction(on the original dataset). The metrics of this model are below:
- **Accuracy**: 86.94%
- **Precision**: 86.82%
- **Recall**: 87.11%
- **AUC-ROC Score**: 86.93%

![Results using VIF Method](https://i.postimg.cc/bNP3NWqg/Screenshot-2024-05-25-at-20-38-57-churn-final-Jupyter-Notebook.png)

![Results using PCA Method](https://i.postimg.cc/MKR0RDpT/Screenshot-2024-05-25-at-20-43-04-churn-final-Jupyter-Notebook.png)

![Results using LDA Method](https://i.postimg.cc/d39sPp93/Screenshot-2024-05-25-at-20-43-45-churn-final-Jupyter-Notebook.png)

![Results using Feature Importance Method](https://i.postimg.cc/B6KyKFLG/Screenshot-2024-05-25-at-20-47-00-churn-final-Jupyter-Notebook.png)

![Results using RFE Method](https://i.postimg.cc/xdyNTxgr/Screenshot-2024-05-25-at-20-48-44-churn-final-Jupyter-Notebook.png)

![Results using Original Dataset Method](https://i.postimg.cc/g25bLmS8/Screenshot-2024-05-25-at-20-49-36-churn-final-Jupyter-Notebook.png)

## Recommendations

- Further data collection and feature engineering could improve model performance.
- Regularly updating the model with new data can help maintain accuracy over time.
- Implementing retention strategies based on model predictions to reduce attrition.

## Limitations

- The dataset may contain biases that could affect the model's predictions.
- The models' performance is limited by the quality and quantity of the available data.

## Future Possibilities of the Project

- Exploring additional algorithms and ensemble methods
- Implementing deep learning models for better performance
- Automating the model updating process with new incoming data
- Developing real-time attrition prediction systems

## References
- [Scikit-learn documentation](https://pypi.org/project/scikit-learn/)
- [XGBoost documentation](https://pypi.org/project/xgboost/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
