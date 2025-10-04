# Lung Cancer Prediction using Machine Learning

This project predicts the risk of lung cancer based on a person’s lifestyle and health symptoms.  
By applying machine learning algorithms, the goal is to identify key risk factors and provide early insights into lung cancer likelihood.

## Author
Keyla Rindani

## Dataset
The dataset used is the Lung Cancer Survey Dataset (`survey lung cancer.csv`), which includes information such as:
- Gender  
- Age  
- Habits (smoking, alcohol consumption, etc)  
- Symptoms (coughing, chest pain, shortness of breath, etc)

All categorical values were encoded numerically:
- YES = 1, NO = 0  
- Male = 1, Female = 0  

## Project Workflow

### 1. Data Preparation
- Imported libraries pandas, numpy, matplotlib, seaborn  
- Loaded the dataset using `pd.read_csv()`  
- Removed duplicates and checked for missing values  

### 2. Data Preprocessing
- Encoded categorical variables using LabelEncoder  
- Normalized binary features to contain only 0 and 1  
- Addressed class imbalance in the target variable before training  

### 3. Exploratory Data Analysis (EDA)
- Visualized the distribution of the target variable (`LUNG_CANCER`)  
- Explored relationships between independent features and the target, including:
  - Smoking habits  
  - Age  
  - Gender influence  

### 4. Modeling
Machine learning models tested:
- Logistic Regression
- KNN
- Random Forest Classifier
- XGBoost 

The data was split into training and testing sets for evaluation.

### 5. Model Evaluation
Models were assessed using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

The Random Forest Classifier achieved the best performance with an accuracy of around 98%.

## Key Insights
- Smoking, Age, and Chest Pain are strong predictors of lung cancer.  
- Random Forest provided stable and accurate results even with a slightly imbalanced dataset.  
- This model can be improved further with a larger and more diverse dataset.

## Tools and Libraries

| Library | Purpose |
|----------|----------|
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Encoding, modeling, and evaluation |
| Jupyter Notebook | Interactive analysis and documentation |

## Next Steps
- Apply hyperparameter tuning (GridSearchCV)  
- Experiment with advanced models such as XGBoost  
- Build a Streamlit or Flask web app for interactive prediction  
- Add clear documentation (README + cleaned notebook) on GitHub  


## Conclusion
This project demonstrates that machine learning can assist in identifying lung cancer risks using simple survey data.  
While not a diagnostic tool, it shows potential for early awareness and risk detection.

*Created by Keyla Rindani – 2025*
