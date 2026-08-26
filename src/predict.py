import pandas as pd
import joblib


# ==================================================
# 1. LOAD TRAINED MODEL
# ==================================================

model_path = "results/groundwater_random_forest.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")


# ==================================================
# 2. TAKE USER INPUT
# ==================================================

print("\n" + "=" * 60)
print("GROUNDWATER AVAILABILITY PREDICTION SYSTEM")
print("=" * 60)

print("\nEnter the groundwater parameters below.\n")


rainfall = float(input("Rainfall (mm): "))

recharge_command = float(
    input("Recharge Worthy Area Command (ha): ")
)

recharge_total = float(
    input("Recharge Worthy Area Total (ha): ")
)

hilly_area = float(
    input("Hilly Area (ha): ")
)

geographical_area = float(
    input("Total Geographical Area (ha): ")
)

groundwater_recharge = float(
    input("Annual Groundwater Recharge (ham): ")
)

environmental_flows = float(
    input("Environmental Flows (ham): ")
)

extractable_resource = float(
    input("Annual Extractable Groundwater Resource (ham): ")
)

extraction_domestic = float(
    input("Domestic Groundwater Extraction (ham): ")
)

extraction_industrial = float(
    input("Industrial Groundwater Extraction (ham): ")
)

extraction_irrigation = float(
    input("Irrigation Groundwater Extraction (ham): ")
)

total_extraction = float(
    input("Total Groundwater Extraction (ham): ")
)

stage_extraction = float(
    input("Stage of Groundwater Extraction (%): ")
)

allocation_domestic = float(
    input("Domestic Allocation 2025 (ham): ")
)


# ==================================================
# 3. CATEGORICAL INPUT
# ==================================================

state = input("State: ")

district = input("District: ")

assessment_unit = input("Assessment Unit: ")


# ==================================================
# 4. CREATE INPUT DATAFRAME
# ==================================================

input_data = pd.DataFrame({
    "Rainfall_mm": [rainfall],
    "Recharge_Worthy_Area_Command_ha": [recharge_command],
    "Recharge_Worthy_Area_Total_ha": [recharge_total],
    "Hilly_Area_ha": [hilly_area],
    "Total_Geographical_Area_ha": [geographical_area],
    "Annual_Ground_Water_Recharge_ham": [groundwater_recharge],
    "Environmental_Flows_ham": [environmental_flows],
    "Annual_Extractable_Ground_Water_Resource_ham": [extractable_resource],
    "Extraction_Domestic_ham": [extraction_domestic],
    "Extraction_Industrial_ham": [extraction_industrial],
    "Extraction_Irrigation_ham": [extraction_irrigation],
    "Total_Ground_Water_Extraction_ham": [total_extraction],
    "Stage_of_Ground_Water_Extraction_pct": [stage_extraction],
    "Allocation_Domestic_2025_ham": [allocation_domestic],
    "STATE": [state],
    "DISTRICT": [district],
    "ASSESSMENT_UNIT": [assessment_unit]
})


# ==================================================
# 5. MAKE PREDICTION
# ==================================================

prediction = model.predict(input_data)


# ==================================================
# 6. DISPLAY RESULT
# ==================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"Predicted Net Groundwater Availability: "
    f"{prediction[0]:,.2f} ham"
)

print(
    f"Equivalent water volume: "
    f"{prediction[0] * 10000:,.2f} m³"
)

print("=" * 60)