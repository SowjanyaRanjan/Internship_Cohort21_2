## Forest Fire Risk Prediction using Machine Learning
Project Overview

Forest fires are one of the major environmental hazards that cause severe damage to forests, wildlife, and human settlements. Early prediction of forest fire risk can help authorities take preventive actions and reduce disaster impact.

This project uses **Machine Learning algorithms** to predict the forest fire risk level (**Low, Moderate, Severe**) using environmental and weather-related features from the **Algerian Forest Fire Dataset**.

The project compares the performance of **Decision Tree** and **Random Forest** classifiers to identify the most accurate model for predicting forest fire risk

Dataset
The dataset used is the **Algerian Forest Fire Dataset**, which contains weather and Fire Weather Index (FWI) related features.

Feature | Description 

Temperature | Ambient temperature 
RH | Relative Humidity
Ws | Wind Speed 
Rain | Rainfall amount 
FFMC | Fine Fuel Moisture Code 
DMC | Duff Moisture Code 
DC | Drought Code 
ISI | Initial Spread Index 
BUI | Build Up Index 
FWI | Fire Weather Index 


**Project Workflow**

1. **Data Collection** – Dataset obtained from Kaggle
2. **Data Cleaning & Preprocessing** – Handling missing values, removing unnecessary columns, feature selection
3. **Exploratory Data Analysis (EDA)** – Understanding feature relationships, visualizing correlations and distributions
4. **Feature Engineering** – Creating additional features such as Heat Index, preparing data for model training
5. **Model Training** – Decision Tree Classifier, Random Forest Classifier
6. **Model Evaluation** – Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, Cross Validation

**Machine Learning Models**

**Decision Tree Classifier**
The Decision Tree algorithm splits the dataset into branches based on feature conditions and predicts the class label.

**Model Performance:**
Accuracy: **87.76%**
F1 Score: **87.76%**
Cross Validation Accuracy: **89.75% ± 7.82%**

The model performs well for Low risk predictions, but struggles to detect Severe fire cases accurately.

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

**Model Performance:**
Accuracy: **93.88%**
F1 Score: **93.94%**
Cross Validation Accuracy: **95.49% ± 4.16%**

Random Forest significantly improves the prediction of Severe fire risk cases and reduces misclassification errors.


**Model Comparison**

Model | Accuracy | CV Accuracy | Severe Class Detection 

Decision Tree = Accuracy 87.76% , CV Acuuracy 89.75% , Severe Class Detection Moderate.
Random Forest = Accuracy 93.88% ,CV Acuuracy 95.49% ,Severe Class Detection Excellent.

**Random Forest performs better** due to ensemble learning and reduced overfitting.

**Key Insights**

**ISI** and **FFMC** are the most important features for predicting forest fire risk.
These two features contribute to about **50% of the predictive power** in the model.
Random Forest shows more **stable cross-validation results**, meaning it generalizes better on unseen data.

**Technologies Used**

**Programming Language:** Python
**Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
**Tools:** Jupyter Notebook, VS Code, GitHub

**Conclusion**

This project demonstrates how Machine Learning can help predict forest fire risk levels using environmental data. Among the tested models, **Random Forest provided the best performance and reliability**, making it suitable for real-world deployment in fire risk monitoring systems.
