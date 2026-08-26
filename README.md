Groundwater Availability Prediction System

A machine learning based system for predicting Net Groundwater Availability for Future Use using the INGRES groundwater assessment dataset.

The system uses a Random Forest Regression model and a Streamlit web application to provide groundwater availability predictions based on automatically retrieved dataset parameters.


Project Overview

This project predicts net groundwater availability for future use based on rainfall, geographical characteristics, groundwater recharge, extraction and other groundwater-related parameters.

Users can select a State, District and Assessment Unit. The application automatically retrieves the corresponding parameters from the dataset and generates a prediction.


Objectives

- Analyze groundwater assessment data
- Perform data preprocessing and cleaning
- Handle missing values and analyze outliers
- Identify important groundwater-related features
- Compare multiple machine learning regression models
- Select the best-performing model
- Predict net groundwater availability for future use
- Develop a user-friendly prediction interface
- Automatically retrieve input parameters from the dataset


Dataset

Source: INGRES Averaged Groundwater Dataset

Records: 742

Original columns: 20

Features used for modelling: 17

Target variable:

Net_GW_Availability_Future_Use_ham

The dataset contains information related to:

- State
- District
- Assessment Unit
- Rainfall
- Recharge-worthy area
- Geographical area
- Groundwater recharge
- Environmental flows
- Extractable groundwater resources
- Domestic groundwater extraction
- Industrial groundwater extraction
- Irrigation groundwater extraction
- Total groundwater extraction
- Stage of groundwater extraction
- Domestic allocation


Machine Learning Workflow

The project follows the following workflow:

Dataset Loading
→ Data Cleaning
→ Missing Value Treatment
→ Outlier Analysis
→ Categorical Encoding
→ Train-Test Split
→ Feature Scaling
→ Feature Correlation Analysis
→ Model Training
→ Model Evaluation
→ Feature Importance Analysis
→ Model Selection
→ Prediction
→ Streamlit Application


Models Evaluated

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor


Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 7393.10 | 11657.29 | 0.8348 |
| Decision Tree | 4966.56 | 9150.93 | 0.8982 |
| Random Forest | 2925.91 | 6197.21 | 0.9533 |
| Gradient Boosting | 3349.96 | 6210.55 | 0.9531 |

Random Forest Regressor was selected as the final model because it achieved the highest R² score and the lowest MAE among the evaluated models.

R² Score: 0.9533

MAE: 2925.91 ham

RMSE: 6197.21 ham


Feature Importance

The most important features identified by the Random Forest model were:

| Feature | Importance |
|---|---:|
| Annual Ground Water Recharge | 0.339517 |
| Stage of Ground Water Extraction | 0.321982 |
| Annual Extractable Ground Water Resource | 0.277359 |
| Environmental Flows | 0.013251 |
| Rainfall | 0.009118 |

These results indicate that groundwater recharge, groundwater extraction stage and extractable groundwater resources have the largest influence on the model predictions.


Application

The project includes a Streamlit-based web application.

The user selects:

- State
- District
- Assessment Unit

The application automatically retrieves the corresponding groundwater and geographical parameters from the INGRES dataset.

The selected parameters are then passed to the trained Random Forest model.

The application displays:

- Selected location
- Retrieved dataset parameters
- Predicted groundwater availability
- Equivalent water volume in cubic metres
- Model R² score
- Reference dataset value
- Absolute difference between prediction and reference value


Technology Stack

Programming Language:
Python

Data Processing:
Pandas
NumPy

Machine Learning:
Scikit-learn
Joblib

Data Visualization:
Matplotlib
Seaborn

Web Application:
Streamlit

Data Support:
CSV
OpenPyXL


Project Structure

INGRES PROJ/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── INGRES_Averaged_Groundwater_Dataset.csv
│
├── results/
│   ├── model_results.csv
│   ├── groundwater_random_forest.pkl
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
└── src/
    ├── preprocessing.py
    ├── model.py
    ├── predict.py
    └── visualization.py


Installation

Create a virtual environment:

python -m venv .venv

Activate the virtual environment.

macOS or Linux:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt


Running the Project

Run the following command from the project root directory:

python -m streamlit run app.py

The application will be available at:

http://localhost:8501


Prediction Process

The application follows this process:

User selects a location
→ Dataset lookup
→ Automatic parameter retrieval
→ Model preprocessing
→ Random Forest prediction
→ Result displayed in Streamlit


Evaluation Metrics

MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values. Lower MAE indicates better performance.

MSE

Mean Squared Error measures the average squared difference between actual and predicted values. Lower MSE indicates better performance.

RMSE

Root Mean Squared Error is the square root of MSE and represents prediction error in the same unit as the target variable. Lower RMSE indicates better performance.

R² Score

R² measures how well the model explains the variation in the target variable. A value closer to 1 indicates better predictive performance.


Results

The Random Forest model achieved an R² score of 0.9533 on the held-out test set.

The model achieved:

R² Score: 0.9533

MAE: 2925.91 ham

RMSE: 6197.21 ham

These results demonstrate strong predictive performance on the evaluated test data.


Future Scope

- Integration with real-time groundwater monitoring sensors
- Integration of time-series groundwater data
- Incorporation of weather and rainfall forecasts
- Geographic visualization using GIS maps
- Automated groundwater risk classification
- Cloud-based deployment
- Periodic model retraining using new groundwater data
- Integration with groundwater management and decision-support systems


Conclusion

This project demonstrates an end-to-end machine learning approach for predicting net groundwater availability for future use.

Multiple regression models were evaluated, with the Random Forest Regressor achieving the best overall performance with an R² score of 0.9533.

The trained model has been integrated into a Streamlit application that automatically retrieves groundwater parameters from the INGRES dataset based on the selected location.

The system provides a practical interface for groundwater availability prediction and demonstrates how machine learning can support data-driven groundwater assessment and management.