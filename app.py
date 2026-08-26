import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Groundwater Availability Predictor",
    page_icon=None,
    layout="wide"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "results/groundwater_random_forest.pkl"
    )


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/INGRES_Averaged_Groundwater_Dataset.csv"
    )


try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Groundwater Availability Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Groundwater Assessment System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "A machine learning system for predicting "
    "Net Groundwater Availability for Future Use."
)


# ============================================================
# NAVIGATION
# ============================================================

tab_prediction, tab_analytics, tab_model, tab_about = st.tabs(
    [
        "Prediction",
        "Analytics",
        "Model Performance",
        "About"
    ]
)


# ============================================================
# PREDICTION PAGE
# ============================================================

with tab_prediction:

    st.header("Groundwater Prediction")

    st.write(
        "Select a location from the dataset. The system will "
        "automatically retrieve the corresponding groundwater "
        "and geographical parameters."
    )

    st.divider()

    # ========================================================
    # LOCATION SELECTION
    # ========================================================

    st.subheader("Location Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        state = st.selectbox(
            "State",
            sorted(
                df["STATE"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:

        districts = sorted(
            df[
                df["STATE"] == state
            ]["DISTRICT"]
            .dropna()
            .unique()
            .tolist()
        )

        district = st.selectbox(
            "District",
            districts
        )

    with col3:

        assessment_units = sorted(
            df[
                (df["STATE"] == state) &
                (df["DISTRICT"] == district)
            ]["ASSESSMENT_UNIT"]
            .dropna()
            .unique()
            .tolist()
        )

        assessment_unit = st.selectbox(
            "Assessment Unit",
            assessment_units
        )

    # ========================================================
    # GET SELECTED ROW FROM DATASET
    # ========================================================

    selected_rows = df[
        (df["STATE"] == state) &
        (df["DISTRICT"] == district) &
        (df["ASSESSMENT_UNIT"] == assessment_unit)
    ]

    if selected_rows.empty:

        st.warning(
            "No dataset record was found for the selected location."
        )

        st.stop()

    selected_row = selected_rows.iloc[0]

    st.divider()

    # ========================================================
    # AUTOMATIC DATA RETRIEVAL
    # ========================================================

    st.subheader("Dataset Parameters")

    st.caption(
        "The following values have been automatically retrieved "
        "from the INGRES groundwater dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("Rainfall and Geographical Parameters")

        st.dataframe(
            pd.DataFrame({
                "Parameter": [
                    "Rainfall (mm)",
                    "Recharge Worthy Area Command (ha)",
                    "Recharge Worthy Area Total (ha)",
                    "Hilly Area (ha)",
                    "Total Geographical Area (ha)"
                ],
                "Value": [
                    selected_row["Rainfall_mm"],
                    selected_row[
                        "Recharge_Worthy_Area_Command_ha"
                    ],
                    selected_row[
                        "Recharge_Worthy_Area_Total_ha"
                    ],
                    selected_row["Hilly_Area_ha"],
                    selected_row[
                        "Total_Geographical_Area_ha"
                    ]
                ]
            }),
            hide_index=True,
            width="stretch"
        )

    with col2:

        st.write("Groundwater Parameters")

        st.dataframe(
            pd.DataFrame({
                "Parameter": [
                    "Annual Ground Water Recharge (ham)",
                    "Environmental Flows (ham)",
                    "Annual Extractable Ground Water Resource (ham)",
                    "Domestic Ground Water Extraction (ham)",
                    "Industrial Ground Water Extraction (ham)",
                    "Irrigation Ground Water Extraction (ham)",
                    "Total Ground Water Extraction (ham)",
                    "Stage of Ground Water Extraction (%)",
                    "Domestic Allocation 2025 (ham)"
                ],
                "Value": [
                    selected_row[
                        "Annual_Ground_Water_Recharge_ham"
                    ],
                    selected_row[
                        "Environmental_Flows_ham"
                    ],
                    selected_row[
                        "Annual_Extractable_Ground_Water_Resource_ham"
                    ],
                    selected_row[
                        "Extraction_Domestic_ham"
                    ],
                    selected_row[
                        "Extraction_Industrial_ham"
                    ],
                    selected_row[
                        "Extraction_Irrigation_ham"
                    ],
                    selected_row[
                        "Total_Ground_Water_Extraction_ham"
                    ],
                    selected_row[
                        "Stage_of_Ground_Water_Extraction_pct"
                    ],
                    selected_row[
                        "Allocation_Domestic_2025_ham"
                    ]
                ]
            }),
            hide_index=True,
            width="stretch"
        )

    st.divider()

    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame({

        "STATE": [selected_row["STATE"]],

        "DISTRICT": [selected_row["DISTRICT"]],

        "ASSESSMENT_UNIT": [
            selected_row["ASSESSMENT_UNIT"]
        ],

        "Rainfall_mm": [
            selected_row["Rainfall_mm"]
        ],

        "Recharge_Worthy_Area_Command_ha": [
            selected_row[
                "Recharge_Worthy_Area_Command_ha"
            ]
        ],

        "Recharge_Worthy_Area_Total_ha": [
            selected_row[
                "Recharge_Worthy_Area_Total_ha"
            ]
        ],

        "Hilly_Area_ha": [
            selected_row["Hilly_Area_ha"]
        ],

        "Total_Geographical_Area_ha": [
            selected_row[
                "Total_Geographical_Area_ha"
            ]
        ],

        "Annual_Ground_Water_Recharge_ham": [
            selected_row[
                "Annual_Ground_Water_Recharge_ham"
            ]
        ],

        "Environmental_Flows_ham": [
            selected_row[
                "Environmental_Flows_ham"
            ]
        ],

        "Annual_Extractable_Ground_Water_Resource_ham": [
            selected_row[
                "Annual_Extractable_Ground_Water_Resource_ham"
            ]
        ],

        "Extraction_Domestic_ham": [
            selected_row[
                "Extraction_Domestic_ham"
            ]
        ],

        "Extraction_Industrial_ham": [
            selected_row[
                "Extraction_Industrial_ham"
            ]
        ],

        "Extraction_Irrigation_ham": [
            selected_row[
                "Extraction_Irrigation_ham"
            ]
        ],

        "Total_Ground_Water_Extraction_ham": [
            selected_row[
                "Total_Ground_Water_Extraction_ham"
            ]
        ],

        "Stage_of_Ground_Water_Extraction_pct": [
            selected_row[
                "Stage_of_Ground_Water_Extraction_pct"
            ]
        ],

        "Allocation_Domestic_2025_ham": [
            selected_row[
                "Allocation_Domestic_2025_ham"
            ]
        ]
    })

    # ========================================================
    # PREDICTION
    # ========================================================

    st.subheader("Generate Prediction")

    predict = st.button(
        "Predict Groundwater Availability",
        type="primary",
        width="stretch"
    )

    if predict:

        try:

            prediction = model.predict(
                input_data
            )[0]

            water_volume = prediction * 10000

            st.success(
                "Prediction generated successfully."
            )

            st.subheader("Prediction Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Net Groundwater Availability",
                    f"{prediction:,.2f} ham"
                )

            with result_col2:

                st.metric(
                    "Equivalent Water Volume",
                    f"{water_volume:,.0f} m³"
                )

            with result_col3:

                st.metric(
                    "Model R² Score",
                    "0.9533"
                )

            st.info(
                f"Location: {assessment_unit}, "
                f"{district}, {state}. "
                f"The Random Forest model predicts "
                f"{prediction:,.2f} ham of net groundwater "
                f"availability for future use."
            )

            # ------------------------------------------------
            # ACTUAL DATASET VALUE
            # ------------------------------------------------

            actual_value = selected_row[
                "Net_GW_Availability_Future_Use_ham"
            ]

            st.subheader(
                "Prediction Validation"
            )

            comparison_col1, comparison_col2 = st.columns(2)

            with comparison_col1:

                st.metric(
                    "Model Prediction",
                    f"{prediction:,.2f} ham"
                )

            with comparison_col2:

                st.metric(
                    "Reference Value",
                    f"{actual_value:,.2f} ham"
                )

            difference = abs(
                actual_value - prediction
            )

            st.write(
                f"Absolute Difference: "
                f"**{difference:,.2f} ham**"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# ============================================================
# ANALYTICS PAGE
# ============================================================

with tab_analytics:

    st.header("Groundwater Analytics")

    st.write(
        "Explore the groundwater dataset, feature relationships "
        "and the factors influencing groundwater availability."
    )

    st.divider()

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Observations", "742")

    with col2:
        st.metric("Input Features", "17")

    with col3:
        st.metric("Target Variable", "Net GW Availability")

    with col4:
        st.metric("Best R² Score", "0.9533")

    st.divider()

    # ========================================================
    # FEATURE-TO-TARGET CORRELATION
    # ========================================================

    st.subheader("Feature-to-Target Correlation")

    target = "Net_GW_Availability_Future_Use_ham"

    numerical_df = df.select_dtypes(
        include=["number"]
    )

    target_corr = (
        numerical_df.corr()[target]
        .drop(target)
        .sort_values(ascending=False)
    )

    st.dataframe(
        target_corr
        .rename("Correlation")
        .round(3)
        .to_frame(),
        width="stretch"
    )

    st.write("")

    st.bar_chart(
        target_corr,
        width="stretch"
    )

    st.caption(
        "Positive values indicate a positive relationship with "
        "net groundwater availability, while negative values "
        "indicate an inverse relationship."
    )

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader("Random Forest Feature Importance")

    importance_data = pd.DataFrame({
        "Feature": [
            "Annual Ground Water Recharge",
            "Stage of Ground Water Extraction",
            "Annual Extractable Ground Water Resource",
            "Environmental Flows",
            "Rainfall",
            "Extraction Irrigation",
            "Domestic Allocation 2025",
            "Total Ground Water Extraction",
            "Extraction Domestic",
            "Recharge Worthy Area Command"
        ],

        "Importance": [
            0.339517,
            0.321982,
            0.277359,
            0.013251,
            0.009118,
            0.006781,
            0.005357,
            0.005345,
            0.003325,
            0.003296
        ]
    })

    importance_data = importance_data.sort_values(
        "Importance",
        ascending=True
    )

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.caption(
        "Feature importance shows how strongly each feature "
        "contributes to the Random Forest model's predictions."
    )

    st.divider()

    # ========================================================
    # CORRELATION MATRIX
    # ========================================================

    st.subheader("Groundwater Correlation Matrix")

    correlation = numerical_df.corr()

    st.dataframe(
        correlation.round(3),
        width="stretch"
    )

    st.caption(
        "Values close to +1 or -1 indicate strong relationships "
        "between variables. Values close to 0 indicate weaker "
        "linear relationships."
    )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

with tab_model:

    st.header("Model Performance")

    st.write(
        "Comparison of the machine learning models evaluated "
        "for groundwater availability prediction."
    )

    st.divider()

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader("Model Comparison")

    results = pd.read_csv(
        "results/model_results.csv"
    )

    st.dataframe(
        results.round(4),
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "R² Score",
            "0.9533"
        )

    with col3:
        st.metric(
            "MAE",
            "2925.91 ham"
        )

    with col4:
        st.metric(
            "RMSE",
            "6197.21 ham"
        )

    st.divider()

    # ========================================================
    # R² COMPARISON
    # ========================================================

    st.subheader("R² Score Comparison")

    r2_data = results[
        ["Model", "R2 Score"]
    ].copy()

    r2_data = r2_data.set_index("Model")

    st.bar_chart(
        r2_data
    )

    st.caption(
        "Higher R² values indicate better explanatory "
        "performance of the regression model."
    )

    st.divider()

    # ========================================================
    # ERROR COMPARISON
    # ========================================================

    st.subheader("RMSE Comparison")

    rmse_data = results[
        ["Model", "RMSE"]
    ].copy()

    rmse_data = rmse_data.set_index("Model")

    st.bar_chart(
        rmse_data
    )

    st.caption(
        "Lower RMSE indicates lower prediction error."
    )

    st.divider()

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader("Actual vs Predicted Values")

    actual_predicted_path = (
        "results/actual_vs_predicted.png"
    )

    try:

        st.image(
            actual_predicted_path,
            width="stretch"
        )

        st.caption(
            "The closer the predicted values are to the "
            "actual values, the better the model performance."
        )

    except Exception:

        st.warning(
            "Actual vs Predicted plot is not available."
        )

    st.divider()

    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader("Selected Model")

    st.success(
        "Random Forest Regressor"
    )

    st.write(
        """
        Random Forest achieved the best overall performance
        among the evaluated models.

        The model achieved an R² score of 0.9533, meaning that
        approximately 95.33% of the variation in net groundwater
        availability is explained by the model's input features.

        It also produced the lowest MAE and a very competitive
        RMSE compared with the other evaluated models.
        """
    )


# ============================================================
# ABOUT PAGE
# ============================================================

with tab_about:

    st.header("About the Project")

    st.write(
        "A machine learning based system for predicting "
        "Net Groundwater Availability for Future Use."
    )

    st.divider()

    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.subheader("Project Objective")

    st.write(
        """
        The objective of this project is to develop a machine
        learning system that predicts the net groundwater
        availability for future use using groundwater,
        rainfall, geographical and extraction-related
        parameters.
        """
    )

    st.divider()

    # ========================================================
    # DATASET
    # ========================================================

    st.subheader("Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Records",
            "742"
        )

    with col2:
        st.metric(
            "Original Features",
            "20"
        )

    with col3:
        st.metric(
            "Model Features",
            "17"
        )

    st.write(
        """
        The dataset contains groundwater assessment records
        covering geographical, rainfall, groundwater recharge,
        groundwater extraction and environmental parameters.

        The target variable used for prediction is:

        Net_GW_Availability_Future_Use_ham
        """
    )

    st.divider()

    # ========================================================
    # MACHINE LEARNING WORKFLOW
    # ========================================================

    st.subheader("Machine Learning Workflow")

    st.write(
        """
        1. Dataset loading

        2. Data preprocessing

        3. Missing value treatment

        4. Outlier analysis

        5. Categorical variable encoding

        6. Feature scaling

        7. Feature correlation analysis

        8. Train-test split

        9. Training multiple regression models

        10. Model evaluation

        11. Feature importance analysis

        12. Selection of the best model

        13. Groundwater availability prediction
        """
    )

    st.divider()

    # ========================================================
    # MODELS
    # ========================================================

    st.subheader("Models Evaluated")

    model_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],

        "R² Score": [
            0.8348,
            0.8982,
            0.9533,
            0.9531
        ],

        "RMSE (ham)": [
            11657.29,
            9150.93,
            6197.21,
            6210.55
        ]
    })

    st.dataframe(
        model_data,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ========================================================
    # SELECTED MODEL
    # ========================================================

    st.subheader("Selected Model")

    st.success(
        "Random Forest Regressor"
    )

    st.write(
        """
        Random Forest was selected because it achieved the
        highest R² score and the lowest MAE among the evaluated
        models.

        Performance:

        R² Score: 0.9533

        MAE: 2925.91 ham

        RMSE: 6197.21 ham
        """
    )

    st.divider()

    # ========================================================
    # IMPORTANT FEATURES
    # ========================================================

    st.subheader("Most Important Features")

    important_features = pd.DataFrame({
        "Feature": [
            "Annual Ground Water Recharge",
            "Stage of Ground Water Extraction",
            "Annual Extractable Ground Water Resource",
            "Environmental Flows",
            "Rainfall"
        ],

        "Importance": [
            0.339517,
            0.321982,
            0.277359,
            0.013251,
            0.009118
        ]
    })

    st.dataframe(
        important_features,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.subheader("Technology Stack")

    st.write(
        """
        Python

        Pandas and NumPy

        Scikit-learn

        Matplotlib and Seaborn

        Joblib

        Streamlit
        """
    )

    st.divider()

    # ========================================================
    # PROJECT OUTPUT
    # ========================================================

    st.subheader("System Output")

    st.write(
        """
        The system accepts groundwater and geographical
        parameters from the user and uses the trained Random
        Forest model to estimate the net groundwater
        availability for future use.

        The predicted value is presented both in ham and
        equivalent cubic metres of water.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Groundwater Availability Prediction System | "
    "Machine Learning Project"
)