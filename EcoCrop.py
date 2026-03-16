import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Load EcoCrop Dataset
# ------------------------------------------------------------

crops = pd.read_csv("EcoCrop_DB.csv", encoding="latin1")

print("Original dataset shape:", crops.shape)

# ------------------------------------------------------------
# 2. Standardize Column Names
# ------------------------------------------------------------

crops.columns = crops.columns.str.strip().str.lower()

# Remove duplicate columns (EcoCrop sometimes has duplicate cliz)
crops = crops.loc[:, ~crops.columns.duplicated()]

print("Columns after cleaning:", len(crops.columns))

print(crops.info())


# ------------------------------------------------------------
# 3. Convert Numeric Columns
# ------------------------------------------------------------

numeric_cols = [
    "topmn","topmx","tmin","tmax",
    "ropmn","ropmx","rmin","rmax",
    "phopmn","phopmx","phmin","phmax",
    "latopmn","latopmx","latmn","latmx",
    "altmx","gmin","gmax"
]
#errors="coerce": If the value cannot be converted to number, replace with NaN
for col in numeric_cols:
    if col in crops.columns:
        crops[col] = pd.to_numeric(crops[col], errors="coerce")


# ------------------------------------------------------------
# 4. Fix Invalid Values
# ------------------------------------------------------------

# Latitude sanity check
crops.loc[(crops["latmn"] > 90) | (crops["latmn"] < -90), "latmn"] = np.nan
crops.loc[(crops["latmx"] > 90) | (crops["latmx"] < -90), "latmx"] = np.nan

# Temperature sanity check
crops.loc[crops["tmin"] < -50, "tmin"] = np.nan
crops.loc[crops["tmax"] > 60, "tmax"] = np.nan


# ------------------------------------------------------------
# 5. Remove Pure Metadata Columns
# ------------------------------------------------------------

metadata_cols = [
    "ecoportcode",
    "auth",
    "famname",
    "syno",
    "comname"
]

crops = crops.drop(columns=[c for c in metadata_cols if c in crops.columns])


# ------------------------------------------------------------
# 6. Minimal Categorical Encoding
# ------------------------------------------------------------

categorical_cols = [
    "lifo",
    "habi",
    "lispa",
    "phys",
    "cat",
    "plat",
    "photo",
    "cliz"
]

#label encoding
for col in categorical_cols:
    if col in crops.columns:
        crops[col] = crops[col].astype("category").cat.codes
# ------------------------------------------------------------
# 6B. Convert Light Tolerance Categories to Numeric Levels
# ------------------------------------------------------------

# EcoCrop light conditions mapped to increasing radiation intensity
light_map = {
    "heavy shade": 1,
    "light shade": 2,
    "cloudy skies": 3,
    "very bright": 4,
    "clear skies": 5
}

for col in ["liopmn","liopmx","limn","limx"]:
    if col in crops.columns:
        crops[col] = crops[col].map(light_map)

# ------------------------------------------------------------
# Remove crops with completely missing light tolerance
# ------------------------------------------------------------

crops = crops.dropna(subset=["liopmn","liopmx"], how="all")


# ------------------------------------------------------------
# Remove crops with completely missing temperature limits
# ------------------------------------------------------------

crops = crops.dropna(subset=["topmn","topmx","tmin","tmax"], how="all")

print("Dataset shape", crops.shape)


# ------------------------------------------------------------
# 10. Data Transformation (Tolerance Ranges)
# ------------------------------------------------------------

# temp_opt_range = width of the ideal temperature window for the crop
# Large window → crop can grow well across a wider temperature variation
# Small window → crop needs very specific temperature conditions
# Helps model understand how temperature-sensitive or tolerant a crop is

crops["temp_opt_range"] = np.abs(crops["topmx"] - crops["topmn"])


# temp_abs_range = total temperature survival range of the crop
# Large range → crop can survive in extreme climates
# Small range → crop survives only in limited climates
# Helps model identify climate-resilient crops for a region

crops["temp_abs_range"] = np.abs(crops["tmax"] - crops["tmin"])


# rain_opt_range = width of optimal rainfall range
# Large range → crop tolerates variable rainfall
# Small range → crop needs precise water conditions
# Helps model match crops to regions with stable or variable rainfall

crops["rain_opt_range"] = np.abs(crops["ropmx"] - crops["ropmn"])


# rain_abs_range = full rainfall survival tolerance
# Large range → crop can survive drought and heavy rainfall
# Small range → crop is sensitive to water stress
# Helps model understand water resilience of crops

crops["rain_abs_range"] = np.abs(crops["rmax"] - crops["rmin"])


# photo_range = daylight hours range suitable for crop growth
# Large range → crop grows under many daylight conditions
# Small range → crop sensitive to seasonal daylight changes
# Helps model determine crop suitability across seasons and latitudes

crops["photo_range"] = np.abs(crops["phopmx"] - crops["phopmn"])


# lat_range = geographic latitude adaptability of the crop
# Large range → crop can grow across many climate zones
# Small range → crop limited to specific geographic regions
# Helps model understand global adaptability of crops

crops["lat_range"] = crops["latmx"] - crops["latmn"]


# light_opt_range = optimal sunlight tolerance window
# Large range → crop tolerates different sunlight levels
# Small range → crop requires specific sunlight intensity
# Helps model match crops with radiation and sunlight availability

crops["light_opt_range"] = np.abs(crops["liopmx"] - crops["liopmn"])


# light_abs_range = full sunlight survival tolerance
# Large range → crop survives in shade and strong sunlight
# Small range → crop sensitive to light stress
# Helps model determine crop survival under varying light conditions

crops["light_abs_range"] = np.abs(crops["limx"] - crops["limn"])


# growth_duration = flexibility in crop growth cycle
# Large duration → crop can adapt to different growing seasons
# Small duration → crop requires a fixed growing period
# Helps model match crops with regional growing season length

crops["growth_duration"] = np.abs(crops["gmax"] - crops["gmin"])


# ------------------------------------------------------------
# 11. Feature Engineering (Midpoints)
# ------------------------------------------------------------

# temp_opt_center = ideal temperature for crop growth
# Value represents the middle of optimal temperature range
# Environment temperature close to this value → high suitability
# Helps model measure how close regional climate is to crop ideal

crops["temp_opt_center"] = (crops["topmn"] + crops["topmx"]) / 2


# rain_opt_center = ideal rainfall requirement of the crop
# Represents the most suitable rainfall condition
# Regions near this value → better crop growth
# Helps model compare regional rainfall with crop preference

crops["rain_opt_center"] = (crops["ropmn"] + crops["ropmx"]) / 2


# photo_center = ideal daylight duration for the crop
# Represents best photoperiod for crop development
# Regions with similar daylight hours → better growth
# Helps model align crop season with regional daylight patterns

crops["photo_center"] = (crops["phopmn"] + crops["phopmx"]) / 2


# light_center = ideal sunlight intensity for crop photosynthesis
# Represents the most suitable light conditions
# Regions close to this value → higher productivity
# Helps model match crop light requirement with solar radiation

crops["light_center"] = (crops["liopmn"] + crops["liopmx"]) / 2

crops.loc[crops["growth_duration"] > 400, "growth_duration"] = np.nan
crops.loc[crops["photo_center"] > 24, "photo_center"] = np.nan
crops.loc[crops["growth_duration"] <= 0, "growth_duration"] = np.nan
# --------------------------------------------------
# Safe CV calculation
# --------------------------------------------------

def coef_variation(series):
    series = series.dropna()
    if series.mean() == 0:
        return np.nan
    return series.std() / series.mean()

# ------------------------------------------------------------
# 13. Final Dataset Overview
# ------------------------------------------------------------

print("\nFinal crop dataset shape:", crops.shape)

print("\nSample rows:")
print(crops.head())
# ------------------------------------------------------------
# Remove variables not related to environmental suitability
# ------------------------------------------------------------

remove_cols = [
    "text", "textr",          # soil texture (not available in env dataset)
    "fer", "ferr",            # soil fertility requirement
    "tox", "toxr",            # soil toxicity tolerance
    "sal", "salr",            # soil salinity tolerance
    "abitol", "abisus",       # abiotic stress descriptions (fire, grazing etc)
    "intri",                  # intrinsic crop properties
    "prosy","latopmn","latopmx"                  # production system (home garden, commercial etc)
]

crops = crops.drop(columns=[c for c in remove_cols if c in crops.columns])



print("\n================ CROP ENVIRONMENTAL REQUIREMENT SUMMARY ================\n")


# --------------------------------------------------
# TEMPERATURE REQUIREMENTS
# --------------------------------------------------

print("---------------- TEMPERATURE REQUIREMENTS ----------------")

print(f"Average ideal temperature for crop growth: {crops['temp_opt_center'].mean():.2f} °C")
print(f"Average optimal temperature tolerance window: {crops['temp_opt_range'].mean():.2f} °C")
print(f"Average total temperature survival range: {crops['temp_abs_range'].mean():.2f} °C")

print("----------------------------------------------------------\n")


# --------------------------------------------------
# RAINFALL REQUIREMENTS
# --------------------------------------------------

print("---------------- RAINFALL REQUIREMENTS ----------------")

print(f"Average ideal rainfall requirement for crops: {crops['rain_opt_center'].mean():.2f} mm")
print(f"Average optimal rainfall tolerance range: {crops['rain_opt_range'].mean():.2f} mm")
print(f"Average total rainfall survival range: {crops['rain_abs_range'].mean():.2f} mm")

print("-------------------------------------------------------\n")


# --------------------------------------------------
# SUNLIGHT REQUIREMENTS
# --------------------------------------------------

print("---------------- SUNLIGHT REQUIREMENTS ----------------")

print(f"Average ideal sunlight level required for photosynthesis: {crops['light_center'].mean():.2f}")
print(f"Average optimal sunlight tolerance range: {crops['light_opt_range'].mean():.2f}")
print(f"Average full sunlight survival tolerance range: {crops['light_abs_range'].mean():.2f}")

print("-------------------------------------------------------\n")


# --------------------------------------------------
# DAYLIGHT REQUIREMENTS
# --------------------------------------------------

print("---------------- DAYLIGHT REQUIREMENTS ----------------")

print(f"Average ideal daylight duration for crop growth: {crops['photo_center'].mean():.2f} hours")
print(f"Average daylight tolerance window across crops: {crops['photo_range'].mean():.2f} hours")

print("-------------------------------------------------------\n")


# --------------------------------------------------
# GROWTH CYCLE
# --------------------------------------------------

print("---------------- GROWTH CYCLE ----------------")

print(f"Average crop growth duration: {crops['growth_duration'].mean():.1f} days")
print(f"Shortest crop growth cycle observed: {crops['growth_duration'].min():.1f} days")
print(f"Longest crop growth cycle observed: {crops['growth_duration'].max():.1f} days")

print("------------------------------------------------\n")


# --------------------------------------------------
# CROP ADAPTABILITY
# --------------------------------------------------

temp_var = coef_variation(crops["temp_opt_range"])
rain_var = coef_variation(crops["rain_opt_range"])
light_var = coef_variation(crops["light_opt_range"])

print("---------------- CROP ADAPTABILITY ----------------")

print(f"Temperature adaptability variability (CV): {temp_var:.2f}")
print(f"Rainfall adaptability variability (CV): {rain_var:.2f}")
print(f"Sunlight adaptability variability (CV): {light_var:.2f}")

print("---------------------------------------------------\n")
# ------------------------------------------------------------
# 14. Save Processed Crop Dataset
# ------------------------------------------------------------

crops.to_parquet(
    "outputs/crop_requirements_features.parquet",
    index=False
)
print(crops.shape)
print("\nSaved: outputs/crop_requirements_features.parquet")

