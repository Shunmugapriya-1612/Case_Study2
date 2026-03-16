import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ------------------------------------------------------------
# 1. Load Environment Dataset
# ------------------------------------------------------------

env_df = pd.read_parquet("outputs/AgroClimate_Feature_Matrix_2025.parquet")



# ------------------------------------------------------------
# 2. Load Crop Requirement Dataset
# ------------------------------------------------------------

crop_df = pd.read_parquet("outputs/crop_requirements_features.parquet")


# ------------------------------------------------------------
# 3. Keep Crop Dataset As-Is (No Dropping / No Imputation)
# ------------------------------------------------------------

crop_model_df = crop_df.copy()

# Keep crop name for reference
crop_model_df["crop_name"] = crop_model_df["scientificname"]





# ------------------------------------------------------------
# 5. Cross Join (Environment × Crop)
# ------------------------------------------------------------
# ------------------------------------------------------------
# STEP 1 — Compute Germany Climate Summary
# ------------------------------------------------------------

temp_min = env_df["Temp"].min()
temp_max = env_df["Temp"].max()

lat_min = env_df["lat"].min()
lat_max = env_df["lat"].max()

growing_season = env_df["Active_GDD_Months"].mean()


# ------------------------------------------------------------
# STEP 2 — Filter Crops Compatible With Germany
# ------------------------------------------------------------

crop_filtered = crop_df.copy()

# Latitude compatibility
crop_filtered = crop_filtered[
    (crop_filtered["latmn"].isna()) |
    (
        (crop_filtered["latmx"] >= lat_min) &
        (crop_filtered["latmn"] <= lat_max)
    )
]

# Temperature survival compatibility
crop_filtered = crop_filtered[
    (crop_filtered["tmax"] >= temp_min) &
    (crop_filtered["tmin"] <= temp_max)
]

# Growing duration compatibility
crop_filtered = crop_filtered[
    (crop_filtered["gmax"].isna()) |
    (crop_filtered["gmax"] <= growing_season * 30)
]

print("Filtered crops:", crop_filtered.shape)

env_df["key"] = 1
crop_filtered["key"] = 1

agro_training_df = env_df.merge(
    crop_filtered,
    on="key"
).drop("key", axis=1)

print("Training dataset size:", agro_training_df.shape)


# ============================================================
# 1. PHOTOPERIOD SUITABILITY
# ============================================================
# Compare daylight hours with crop photoperiod requirement

agro_training_df["Photoperiod_Suitable"] = (
    abs(
        agro_training_df["Daylight_hours"] -
        agro_training_df["photo_center"]
    ) <= (agro_training_df["photo_range"] / 2 + 4)
)

# ============================================================
# 5. SEASONAL LIGHT AVAILABILITY
# ============================================================

agro_training_df["Seasonal_Light_Suitable"] = (
    (agro_training_df["Above_Annual_Mean"] == 1) |
    (agro_training_df["Optimal_hours"] >= 6)
)

# ============================================================
# 8. FINAL LIGHT SUITABILITY LABEL
# ============================================================

agro_training_df["Light_Suitable"] = (
    (agro_training_df["DLI_value"] >= 8) &
    (
        agro_training_df["Photoperiod_Suitable"] |
        agro_training_df["Seasonal_Light_Suitable"]
    )
).astype(int)

# ============================================================
# 9. REMOVE INTERMEDIATE VARIABLES
# ============================================================

agro_training_df.drop(columns=[
    "Photoperiod_Suitable",
    "Seasonal_Light_Suitable",
], inplace=True)

# ============================================================
# 1. AIR TEMPERATURE SUITABILITY
# ============================================================

# Optimal air temperature range for crop growth
agro_training_df["AirTemp_Optimal"] = (
    (agro_training_df["Temp"] >= agro_training_df["topmn"]) &
    (agro_training_df["Temp"] <= agro_training_df["topmx"])
)

# Survival temperature limits
agro_training_df["AirTemp_Survival"] = (
    (agro_training_df["Temp"] >= agro_training_df["tmin"]) &
    (agro_training_df["Temp"] <= agro_training_df["tmax"])
)

# Avoid extreme heat or frost
agro_training_df["AirTemp_NoStress"] = (
    (agro_training_df["Heat_Stress"] == 0) &
    (agro_training_df["Cold_Stress"] == 0)
)

# Seasonal temperature anomaly should not be extreme
agro_training_df["AirTemp_Anomaly_OK"] = (
    agro_training_df["Temp_Anomaly"].abs() < 10
)

# Final air temperature suitability
agro_training_df["AirTemp_Suitable"] = (
    agro_training_df["AirTemp_Survival"] &
    agro_training_df["AirTemp_NoStress"] &
    (
        agro_training_df["AirTemp_Optimal"] |
        agro_training_df["AirTemp_Anomaly_OK"]
    )
).astype(int)

# ============================================================
# 2. SOIL TEMPERATURE SUITABILITY
# ============================================================

# Root zone temperature within crop optimal range
agro_training_df["RootTemp_Optimal"] = (
    (agro_training_df["SoilTemp_L2"] >= agro_training_df["topmn"]) &
    (agro_training_df["SoilTemp_L2"] <= agro_training_df["topmx"])
)

# Surface temperature for germination
agro_training_df["SurfaceTemp_OK"] = (
    (agro_training_df["SoilTemp_L1"] >= agro_training_df["topmn"] - 5) &
    (agro_training_df["SoilTemp_L1"] <= agro_training_df["topmx"] + 5)
)

# Soil temperature stability
agro_training_df["SoilTemp_Stable"] = (
    (agro_training_df["SoilTemp_L2_STD"] < 8) &
    (agro_training_df["SoilTemp_L2_CV"] < 1.5)
)

# Avoid extreme air temperature effects on soil
agro_training_df["SoilTemp_NoExtremeAir"] = (
    (agro_training_df["Heat_Stress"] == 0) &
    (agro_training_df["Cold_Stress"] == 0)
)

# Final soil temperature suitability
agro_training_df["SoilTemp_Suitable"] = (
    agro_training_df["SoilTemp_NoExtremeAir"] &
    (
        agro_training_df["RootTemp_Optimal"] |
        agro_training_df["SurfaceTemp_OK"]
    ) &
    agro_training_df["SoilTemp_Stable"]
).astype(int)

# ============================================================
# 3. SOIL MOISTURE SUITABILITY
# ============================================================

# Root zone moisture optimal range
agro_training_df["RootMoisture_Optimal"] = (
    (agro_training_df["SWVL2"] >= 0.20) &
    (agro_training_df["SWVL2"] <= 0.40)
)

# Root moisture survival limits
agro_training_df["RootMoisture_Survival"] = (
    (agro_training_df["SWVL2"] >= 0.10) &
    (agro_training_df["SWVL2"] <= 0.60)
)

# Avoid drought
agro_training_df["No_Deep_Drought"] = (
    agro_training_df["Deep_Drought_Flag"] == 0
)

# Moisture stability
agro_training_df["Moisture_Stable"] = (
    agro_training_df["SWVL2_CV"] < 0.6
)

# Vertical soil moisture balance
agro_training_df["Vertical_Moisture_OK"] = (
    agro_training_df["Vertical_Moisture_Gradient"].abs() < 0.25
)

# Moisture anomaly check
agro_training_df["Moisture_Anomaly_OK"] = (
    agro_training_df["SWVL2_Anomaly"] > -0.15
)

# Joint temperature-moisture suitability
agro_training_df["Joint_Temp_Moisture"] = (
    agro_training_df["Joint_Suitability_Flag"] == 1
)

# Final soil moisture suitability
agro_training_df["SoilMoisture_Suitable"] = (
    agro_training_df["RootMoisture_Survival"] &
    agro_training_df["No_Deep_Drought"] &
    agro_training_df["Moisture_Stable"] &
    (
        agro_training_df["RootMoisture_Optimal"] |
        agro_training_df["Moisture_Anomaly_OK"]
    ) &
    (
        agro_training_df["Vertical_Moisture_OK"] |
        agro_training_df["Joint_Temp_Moisture"]
    )
).astype(int)
# ============================================================
# 4. FINAL CROP SUITABILITY LABEL
# ============================================================

agro_training_df["Crop_Suitability_Percent"] = (
    0.35 * agro_training_df["Light_Suitable"] +
    0.30 * agro_training_df["AirTemp_Suitable"] +
    0.20 * agro_training_df["SoilTemp_Suitable"] +
    0.15 * agro_training_df["SoilMoisture_Suitable"]
) * 100

agro_training_df["Crop_Suitable"] = (
    agro_training_df["Crop_Suitability_Percent"] >= 50
).astype(int)


# ============================================================
# 5. REMOVE INTERMEDIATE VARIABLES
# ============================================================

agro_training_df.drop(columns=[
    "AirTemp_Optimal",
    "AirTemp_Survival",
    "AirTemp_NoStress",
    "AirTemp_Anomaly_OK",
    "RootTemp_Optimal",
    "SurfaceTemp_OK",
    "SoilTemp_Stable",
    "SoilTemp_NoExtremeAir",
    "RootMoisture_Optimal",
    "RootMoisture_Survival",
    "No_Deep_Drought",
    "Moisture_Stable",
    "Vertical_Moisture_OK",
    "Moisture_Anomaly_OK",
    "Joint_Temp_Moisture"
], inplace=True, errors="ignore")


# ============================================================
# 6. CHECK FINAL DISTRIBUTION
# ============================================================

print("\nSuitability Percent Distribution\n")
print(agro_training_df["Crop_Suitability_Percent"].describe())


# ============================================================
# FEATURE SELECTION
# ============================================================

drop_cols = [

    # identifiers
    "crop_name",
    "scientificname",
    "time",

    # labels
    "Crop_Suitable",
    "Crop_Suitability_Percent",

    # rule outputs (avoid leakage)
    "Light_Suitable",
    "AirTemp_Suitable",
    "SoilTemp_Suitable",
    "SoilMoisture_Suitable",

    # rule helper variables
    "Optimal_hours",
    "Suitable_Low",
    "Suitable_Mid",
    "Suitable_High"
]

X = agro_training_df.drop(columns=[c for c in drop_cols if c in agro_training_df.columns])

# IMPORTANT CHANGE → regression target
y = agro_training_df["Crop_Suitability_Percent"]

print("Feature matrix:", X.shape)


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

soil_depth_map = {
    "shallow (20-50 cm)": 1,
    "medium (50-150 cm)": 2,
    "deep (>>150 cm)": 3
}

X["dep"] = X["dep"].map(soil_depth_map)
X["depr"] = X["depr"].map(soil_depth_map)

drainage_map = {
    "poorly (saturated >50% of year)": 1,
    "well (dry spells)": 2,
    "excessive (dry/moderately dry)": 3,
    "well (dry spells), excessive (dry/moderately dry)": 3,
    "poorly (saturated >50% of year), well (dry spells)": 2,
    "poorly (saturated >50% of year), well (dry spells), excessive (dry/moderately dry)": 2
}

X["dra"] = X["dra"].map(drainage_map)
X["drar"] = X["drar"].map(drainage_map)

X["Solar_Regime"] = X["Solar_Regime"].astype("category").cat.codes

# handle missing values
X = X.fillna(-1)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# MODEL ( XGBRegressor)
# ============================================================



model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

pred = model.predict(X_test)


# ============================================================
# REGRESSION METRICS
# ============================================================



print("\nRegression Metrics\n")

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop Important Features\n")

print(importance.head(20))

# ============================================================
# 1. PREDICT SUITABILITY FOR ENTIRE DATASET
# ============================================================

print("\nGenerating suitability predictions...\n")

agro_training_df["Predicted_Suitability"] = model.predict(X)


agro_training_df["Predicted_Suitability"] = (
    agro_training_df["Predicted_Suitability"]
    .clip(0,100)
)

# ============================================================
# 2. CREATE SMALL DATAFRAME FOR RANKING
# (reduces memory usage)
# ============================================================

rank_df = agro_training_df[
    ["lat", "lon", "time", "scientificname", "Predicted_Suitability"]
].copy()


# ============================================================
# 3. RANK CROPS PER LOCATION AND MONTH
# ============================================================

print("Ranking crops for each location and month...\n")
rank_df["month"] = rank_df["time"].dt.month
rank_df["rank"] = (
    rank_df
    .groupby(["lat", "lon", "month"])["Predicted_Suitability"]
    .rank(ascending=False, method="first")
)


# ============================================================
# 4. SELECT TOP N CROPS
# ============================================================

TOP_N = 3

top_crops = rank_df[
    rank_df["rank"] <= TOP_N
].copy()


# ============================================================
# 5. SORT RESULTS
# ============================================================

top_crops = top_crops.sort_values(
    ["lat", "lon", "month", "rank"]
)


# ============================================================
# 6. CREATE RECOMMENDATION TABLE
# ============================================================

print("Creating recommendation summary...\n")

recommendations = (
    top_crops
    .groupby(["lat", "lon", "month"])
    .apply(
        lambda x: ", ".join(
            x["scientificname"] + " (" +
            x["Predicted_Suitability"].round(1).astype(str) + "%)"
        )
    )
    .reset_index(name="Top_Crops")
)


# ============================================================
# 7. DISPLAY SAMPLE OUTPUT
# ============================================================

print("\nSample Crop Recommendations\n")

print(recommendations.head(20))
print(recommendations.shape)

# ============================================================
# 8. SAVE OUTPUT FILES
# ============================================================

recommendations.to_parquet(
    "outputs/crop_recommendations.parquet",
    index=False
)

top_crops.to_parquet(
    "outputs/top_crop_rankings.parquet",
    index=False
)

print("\nFiles saved:")
print("outputs/crop_recommendations.parquet")
print("outputs/top_crop_rankings.parquet")