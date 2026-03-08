# =============================================================================================================
#                                   PHASE 1 — Data Loading & Regional Subsetting
# =============================================================================================================
import xarray as xr
import numpy as np
import pandas as pd
import glob


#Restricting the regions to Germany
LAT_MIN, LAT_MAX = 47, 55
LON_MIN, LON_MAX = 5, 15

# Folder where monthly data are stored
files = sorted(glob.glob("ST/*.nc"))

print("Found files:", len(files))

datasets = []

for f in files:
    print("Processing:", f)

    ds = xr.open_dataset(
        f,
        engine="netcdf4",  # important for ERA5 GRIB-converted files
    )

    # Subset Germany immediately
    ds = ds.sel(
        latitude=slice(LAT_MAX, LAT_MIN),  # latitude reversed
        longitude=slice(LON_MIN, LON_MAX)
    )
# =========================================================================================================
#                             PHASE 2 — Data Transformation
# =========================================================================================================

    # Convert hourly → monthly mean for each month, so that like Solar Data frame there is one value for each month
    ds_month = ds.resample(valid_time="1ME").mean()
    # Extract variables like t2m - air Temperature, swvl1,swvl2 - soil moisture of layer 1 and 2 and stl1,stl2 - soil Temperature of layer 1 and 2
    ds_month = ds_month[["t2m", "stl1", "stl2", "swvl1", "swvl2"]]
    #xarray individual monthly dataset is then appended to list - because to carry out above data transformation and appending to list is memory effecient and faster
    datasets.append(ds_month)

# Concatenate all months in timely dimensions
ds_all = xr.concat(datasets, dim="valid_time")

#Renaming the columns as similar to that of Solar Features Data Frame
ds_all = ds_all.rename({
    "valid_time": "time",
    "latitude": "lat",
    "longitude": "lon"
})

#sort by months so to perform analysis on data from winter to summer 
ds_all = ds_all.sortby("time")



#droping the unwanted co-ordinates other than time, lat and lon
ds_all = ds_all.drop_vars("number", errors="ignore")


# =============================================================================
#                            PHASE 2.5 — Data Cleaning
# =============================================================================

# Drop spatial points (lat/lon) that are NaN (e.g., water bodies)
# We check the subset of your 5 variables. If any are NaN at a location, that point is dropped.
# a region's(LAT and LON) data is removed  whose 't2m', 'stl1', 'stl2', 'swvl1', 'swvl2' are empty 
ds_all = ds_all.dropna(dim="lat", how="any", subset=["t2m", "stl1", "stl2", "swvl1", "swvl2"])
ds_all = ds_all.dropna(dim="lon", how="any", subset=["t2m", "stl1", "stl2", "swvl1", "swvl2"])

print("Cleaned Dataset Dimensions:", ds_all.dims)

# =============================================================================================================
#                            PHASE 3 — Feature Engineering (Temperature + Soil)
# =============================================================================================================
# As Crops respond to °C thresholds, converting to celsius.
#convert the temperature from Kelvin to Celsius
ds_all["t2m"] = ds_all["t2m"] - 273.15
# Convert soil temperature 1 from Kelvin to Celsius
ds_all["stl1"] = ds_all["stl1"] - 273.15
# Convert soil temperature 2 from Kelvin to Celsius
ds_all["stl2"] = ds_all["stl2"] - 273.15

# Define soil layers explicitly
soil_temp_l1 = ds_all["stl1"]   # 0–7 cm
soil_temp_l2 = ds_all["stl2"]   # 7–28 cm   

#Extracting Temperature array based on time*lat*lon, required for next steps
temp = ds_all["t2m"]

# ============================================================
# Growing Degree Days (Base 10°C)
# ============================================================

BASE_TEMP = 10
days_in_month = ds_all.time.dt.days_in_month
#How much heat above the growth threshold accumulated during the season?
# if jan's temp > 0, 
# for eg jan's temp = 11, 
# then temp> BASE_TEMP = True 
# gdd = 11 - BASE_TEMP = 1
gdd_monthly = xr.where(
    temp > BASE_TEMP,
    (temp - BASE_TEMP) * days_in_month,
    0
)

#GDD predicts crop feasibility. GDD understand which region has long summer vs short summer
gdd_annual = gdd_monthly.sum(dim="time")





# ============================================================
# Growing Season Length (>10°C)
# ============================================================
# For how many months was growth possible?
gdd_active_flag = (temp > 10).astype(int)
gdd_active_months = gdd_active_flag.sum(dim="time")

# Calculate Temperature for each Latitude and longitude, not based on each month, basically time co-ordinate is ignored - Dimensionality Reduction.  
#Temperature Change - Temp gradient - how is the this month's temperature compared to previous month for each latitude and longitude
# Was it Rapid warming (spring) or Rapid cooling (autumn)
temp_gradient = temp.diff(dim="time").bfill(dim="time")

#Annual temperature mean for each Latitude and longitude, some crops perfer cool temperature and some may hot temperature.
annual_temp_mean = temp.mean(dim="time")

#Is this month warmer or cooler than usual for this location? 
#Anomaly=Tmonth − Tannual
temp_anomaly = temp - annual_temp_mean

#this flag captures extreme heat risk
#1 if temperature > 35°C , 0 otherwise
heat_stress = (temp > 35).astype(int)

#this flag captures extreme cold risk 
#1 if temperature < 5°C, 0 otherwise
cold_stress = (temp < 5).astype(int)

#Soil moisture
# ============================================================
# Vertical Moisture Structure
# ============================================================
#Is the surface wetter or is the subsurface wetter?
# Difference between surface and deeper soil moisture.
# Positive value → surface wetter than subsoil.
# Negative value → deeper soil stores more water (drought buffering).
vertical_gradient = ds_all["swvl1"] - ds_all["swvl2"]

# ============================================================
# Stability & Variability Metrics
# ============================================================

# ------------------------------------------------------------
# Month-to-Month Change (Drying / Wetting Speed)
# ------------------------------------------------------------
# Measures how quickly soil moisture changes between consecutive months.
# Positive value → soil is getting wetter (rain recharge).
# Negative value → soil is drying (evaporation or plant uptake).
# Useful to detect drought onset or rapid soil drying that may damage crops.
#To identify the rate of drying or wetting.
swvl1_gradient = ds_all["swvl1"].diff(dim="time").bfill(dim="time")
swvl2_gradient = ds_all["swvl2"].diff(dim="time").bfill(dim="time")

# ------------------------------------------------------------
# Annual Mean Soil Moisture (Long-Term Baseline)
# ------------------------------------------------------------
# Represents the typical moisture level at each location over the year.
# High value → generally moist region.
# Low value → chronically dry region.
# Useful to determine whether crops requiring moist or dry soils are suitable.
#Annual soil Moisture mean for each Latitude and longitude
swvl1_mean = ds_all["swvl1"].mean(dim="time")
swvl2_mean = ds_all["swvl2"].mean(dim="time")

# ------------------------------------------------------------
# Temporal Variability (Standard Deviation)
# ------------------------------------------------------------
# Measures how strongly soil moisture fluctuates through the year.
# High value → strong wet–dry cycles.
# Low value → stable moisture conditions.
# It tells How stable the soil water supply is.
swvl1_std = ds_all["swvl1"].std(dim="time", skipna=True)
swvl2_std = ds_all["swvl2"].std(dim="time", skipna=True)

# ------------------------------------------------------------
# Soil Moisture Stability (Layer 1 & Layer 2)
# ------------------------------------------------------------

# Coefficient of Variation (CV) for surface soil moisture.
# It measures how unstable the water supply is compared to its average level.
# High CV → soil moisture changes a lot during the year.
# Low CV → soil moisture is stable and predictable.
# Stable moisture is better for crops that are sensitive to drought stress.
swvl1_cv = swvl1_std / (swvl1_mean + 1e-6).fillna(0)

# Important for deep-root crops.
# Coefficient of Variation (CV) for surface soil moisture.
# It measures how unstable the water supply is compared to its average level.
# High CV → soil moisture changes a lot during the year.
# Low variability means deeper roots have a reliable water supply.
swvl2_cv = swvl2_std / (swvl2_mean + 1e-6).fillna(0)

# ------------------------------------------------------------
# Soil Moisture Anomaly (Departure from Normal)
# ------------------------------------------------------------
# Shows whether a specific month is wetter or drier than the typical annual condition.
# Positive value → wetter than normal.
# Negative value → drier than normal.
# Helps identify drought stress or excess moisture events during the growing season.
#Is this month wetter or drier than normal for this location?
swvl1_anomaly = ds_all["swvl1"] - swvl1_mean
swvl2_anomaly = ds_all["swvl2"] - swvl2_mean

# ------------------------------------------------------------
# Air Temperature Stability (Crop Climate Risk Indicator)
# ------------------------------------------------------------

# Measures how much air temperature changes throughout the year.
# High value → strong seasonal swings (hot summers, cold winters).
# Low value → more stable temperature conditions.
# Crop relevance: High fluctuation increases heat and frost stress risk.
temp_std = temp.std(dim="time", skipna=True)

# It shows how unstable the climate is compared to how warm the region normally is.
# High CV → unstable climate, higher thermal stress risk for crops.
# Low CV → predictable growing environment.
temp_cv = temp_std / (annual_temp_mean + 1e-6).fillna(0)

# Difference between hottest and coldest months.
# Large amplitude → strong seasonal contrast.
# Crop relevance: Needed for seasonal crops, but very large range
# increases frost and heat stress risk.
temp_amplitude = temp.max(dim="time") - temp.min(dim="time")

# ------------------------------------------------------------
# Surface Soil Temperature Stability (Germination Zone)
# ------------------------------------------------------------

# Measures how much surface soil temperature fluctuates.
# Important for seed germination and early root development.
# High variation → possible stress for young plants.
soil_temp_l1_std = soil_temp_l1.std(dim="time", skipna=True)
# Represents typical thermal condition of the topsoil.
soil_temp_l1_mean = soil_temp_l1.mean(dim="time")

#How unstable the surface soil temperature is across the entire year compared to its typical average level.
# High CV → unstable germination environment.
# Low CV → stable conditions for early crop growth.
soil_temp_l1_cv = soil_temp_l1_std / (soil_temp_l1_mean + 1e-6).fillna(0)



# ------------------------------------------------------------
# Surface Soil Temperature Stability (Germination Zone)
# ------------------------------------------------------------

# Measures how much deeper soil temperature fluctuates across the year.
# Calculated per region using monthly values.
# High value → large seasonal temperature swings in the root zone.
# Low value → stable deeper soil temperature
soil_temp_l2_std = soil_temp_l2.std(dim="time", skipna=True)
#What is the average temperature that crop roots experience during the year in this region?
soil_temp_l2_mean = soil_temp_l2.mean(dim="time")

#How large the temperature fluctuations are relative to the normal (average) root-zone temperature
# High value → root temperature changes a lot during the year.
# Low value → root temperature stays more steady during the year.
soil_temp_l2_cv = soil_temp_l2_std / (soil_temp_l2_mean + 1e-6).fillna(0)

# ============================================================
# Soil–Air Coupling Index
# ============================================================

#Marks months when deeper soil moisture is critically low.
drought_flag_deep = (ds_all["swvl2"] < 0.15).astype(int)

#Finds whether drought lasted for consecutive months in a region.
#Short drought → crops may recover
#Long drought → crop failure risk
drought_persistence = (
    drought_flag_deep
        .rolling(time=2)
        .sum()
        .max(dim="time")
)

# Counts how many months were extremely hot or cold
#Too many heat months → heat stress risk
#Too many cold months → frost damage risk
heat_months = heat_stress.sum(dim="time")
cold_months = cold_stress.sum(dim="time")

# Measures how many months provide ideal growth temperature.
optimal_temp_flag = ((temp >= 18) & (temp <= 30)).astype(int)


# Measures how often roots receive adequate water — not too dry, not waterlogged.
optimal_min = 0.20
optimal_max = 0.40

optimal_moisture_flag = (
    (ds_all["swvl2"] >= optimal_min) &
    (ds_all["swvl2"] <= optimal_max)
).astype(int)

optimal_moisture_months = optimal_moisture_flag.sum(dim="time")

# Counts months when both temperature AND moisture are suitable at the same time.
joint_suitability_flag = (
    optimal_temp_flag &
    optimal_moisture_flag
)
# ================================================================================================================
#                               PHASE 4 — Dimensional Reduction (3D → Tabular)
# ================================================================================================================

# ============================================================
# AIR TEMPERATURE FEATURES (Monthly)
# ============================================================

temp_df = temp.to_dataframe(name="Temp").reset_index()

grad_df = temp_gradient.to_dataframe(
    name="Temp_Gradient"
).reset_index()

anom_df = temp_anomaly.to_dataframe(
    name="Temp_Anomaly"
).reset_index()

heat_df = heat_stress.to_dataframe(
    name="Heat_Stress"
).reset_index()

cold_df = cold_stress.to_dataframe(
    name="Cold_Stress"
).reset_index()

optimal_temp_df = optimal_temp_flag.to_dataframe(
    name="Optimal_Temp_Flag"
).reset_index()


# ============================================================
# SOIL TEMPERATURE FEATURES (Monthly)
# ============================================================

soil_temp_l1_df = soil_temp_l1.to_dataframe(
    name="SoilTemp_L1"
).reset_index()

soil_temp_l2_df = soil_temp_l2.to_dataframe(
    name="SoilTemp_L2"
).reset_index()


# ============================================================
# SOIL MOISTURE FEATURES (Monthly)
# ============================================================

swvl1_df = ds_all["swvl1"].to_dataframe(
    name="SWVL1"
).reset_index()

swvl2_df = ds_all["swvl2"].to_dataframe(
    name="SWVL2"
).reset_index()

swvl1_grad_df = swvl1_gradient.to_dataframe(
    name="SWVL1_Gradient"
).reset_index()

swvl2_grad_df = swvl2_gradient.to_dataframe(
    name="SWVL2_Gradient"
).reset_index()

swvl1_anom_df = swvl1_anomaly.to_dataframe(
    name="SWVL1_Anomaly"
).reset_index()

swvl2_anom_df = swvl2_anomaly.to_dataframe(
    name="SWVL2_Anomaly"
).reset_index()

vertical_grad_df = vertical_gradient.to_dataframe(
    name="Vertical_Moisture_Gradient"
).reset_index()

deep_drought_df = drought_flag_deep.to_dataframe(
    name="Deep_Drought_Flag"
).reset_index()

optimal_moisture_df = optimal_moisture_flag.to_dataframe(
    name="Optimal_Deep_Moisture_Flag"
).reset_index()

joint_df = joint_suitability_flag.to_dataframe(
    name="Joint_Suitability_Flag"
).reset_index()


# ============================================================
# MERGE ALL MONTHLY FEATURES
# ============================================================

climate_features = (
    temp_df
    .merge(grad_df, on=["time","lat","lon"], how="left")
    .merge(anom_df, on=["time","lat","lon"], how="left")
    .merge(heat_df, on=["time","lat","lon"], how="left")
    .merge(cold_df, on=["time","lat","lon"], how="left")
    .merge(optimal_temp_df, on=["time","lat","lon"], how="left")

    .merge(soil_temp_l1_df, on=["time","lat","lon"], how="left")
    .merge(soil_temp_l2_df, on=["time","lat","lon"], how="left")

    .merge(swvl1_df, on=["time","lat","lon"], how="left")
    .merge(swvl2_df, on=["time","lat","lon"], how="left")
    .merge(swvl1_grad_df, on=["time","lat","lon"], how="left")
    .merge(swvl2_grad_df, on=["time","lat","lon"], how="left")
    .merge(swvl1_anom_df, on=["time","lat","lon"], how="left")
    .merge(swvl2_anom_df, on=["time","lat","lon"], how="left")

    .merge(vertical_grad_df, on=["time","lat","lon"], how="left")
    .merge(deep_drought_df, on=["time","lat","lon"], how="left")
    .merge(optimal_moisture_df, on=["time","lat","lon"], how="left")
    .merge(joint_df, on=["time","lat","lon"], how="left")
)


# ============================================================
# REGIONAL / ANNUAL FEATURES (lat-lon level)
# ============================================================

gdd_df = gdd_annual.to_dataframe(
    name="GDD_Annual"
).reset_index()

gdd_months_df = gdd_active_months.to_dataframe(
    name="Active_GDD_Months"
).reset_index()

swvl2_mean_df = swvl2_mean.to_dataframe(
    name="SWVL2_Mean"
).reset_index()

swvl2_cv_df = swvl2_cv.to_dataframe(
    name="SWVL2_CV"
).reset_index()

drought_persistence_df = drought_persistence.to_dataframe(
    name="Deep_Drought_Persistence"
).reset_index()


# ============================================================
# TEMPERATURE STABILITY FEATURES
# ============================================================

temp_std_df = temp_std.to_dataframe(
    name="Temp_STD"
).reset_index()

temp_cv_df = temp_cv.to_dataframe(
    name="Temp_CV"
).reset_index()

temp_amp_df = temp_amplitude.to_dataframe(
    name="Temp_Amplitude"
).reset_index()


# ============================================================
# SOIL TEMPERATURE STABILITY FEATURES
# ============================================================

soil_temp_l1_std_df = soil_temp_l1_std.to_dataframe(
    name="SoilTemp_L1_STD"
).reset_index()

soil_temp_l1_cv_df = soil_temp_l1_cv.to_dataframe(
    name="SoilTemp_L1_CV"
).reset_index()

soil_temp_l2_std_df = soil_temp_l2_std.to_dataframe(
    name="SoilTemp_L2_STD"
).reset_index()

soil_temp_l2_cv_df = soil_temp_l2_cv.to_dataframe(
    name="SoilTemp_L2_CV"
).reset_index()


# ============================================================
# MERGE REGIONAL FEATURES INTO MONTHLY DATASET
# ============================================================

climate_features = (
    climate_features
    .merge(gdd_df, on=["lat","lon"], how="left")
    .merge(gdd_months_df, on=["lat","lon"], how="left")

    .merge(swvl2_mean_df, on=["lat","lon"], how="left")
    .merge(swvl2_cv_df, on=["lat","lon"], how="left")
    .merge(drought_persistence_df, on=["lat","lon"], how="left")

    .merge(temp_std_df, on=["lat","lon"], how="left")
    .merge(temp_cv_df, on=["lat","lon"], how="left")
    .merge(temp_amp_df, on=["lat","lon"], how="left")

    .merge(soil_temp_l1_std_df, on=["lat","lon"], how="left")
    .merge(soil_temp_l1_cv_df, on=["lat","lon"], how="left")

    .merge(soil_temp_l2_std_df, on=["lat","lon"], how="left")
    .merge(soil_temp_l2_cv_df, on=["lat","lon"], how="left")
)

print("Final climate matrix shape:", climate_features.shape)
# ================================================================================================================
#                                      PHASE 5 — AGRO-CLIMATE SUMMARY
# ================================================================================================================
# ------------------------------------------------------------
# DATASET STRUCTURE
# ------------------------------------------------------------
print("---------------- DATASET STRUCTURE ----------------")

n_time = ds_all.sizes["time"]
n_lat = ds_all.sizes["lat"]
n_lon = ds_all.sizes["lon"]

print(f"Time steps (months): {n_time}")
print(f"Latitude grid points: {n_lat}")
print(f"Longitude grid points: {n_lon}")

total_cells = n_lat * n_lon
print(f"Total spatial grid cells: {total_cells}")
print("--------------------------------------------------\n")


# ------------------------------------------------------------
# TEMPERATURE REGIME
# ------------------------------------------------------------
print("---------------- TEMPERATURE REGIME ----------------")

jan_temp = float(temp.isel(time=0).mean())
jul_temp = float(temp.isel(time=6).mean())

print(f"Average air temperature in January : {jan_temp:.2f} °C")
print(f"Average air temperature in July    : {jul_temp:.2f} °C")

annual_mean = float(temp.mean().values)
print(f"Mean annual air temperature across Germany: {annual_mean:.2f} °C")

print("------------------------------------------------------\n")


# ------------------------------------------------------------
# GROWING SEASON ANALYSIS
# ------------------------------------------------------------
print("---------------- GROWING SEASON ----------------")

avg_gdd = float(gdd_annual.mean())
gdd_min = float(gdd_annual.min())
gdd_max = float(gdd_annual.max())

print(f"Average Growing Degree Days (base 10°C): {avg_gdd:.0f}")
print(f"GDD range across Germany: {gdd_min:.0f} – {gdd_max:.0f}")

avg_growing_months = float(gdd_active_months.mean())

print(f"Average growing season length: {avg_growing_months:.1f} months")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# SEASONAL TRANSITION SPEED
# ------------------------------------------------------------
print("---------------- SEASONAL TRANSITIONS ----------------")

max_rise = float(temp_gradient.max())
max_drop = float(temp_gradient.min())

print(f"Fastest warming rate : {max_rise:.2f} °C/month")
print(f"Fastest cooling rate : {max_drop:.2f} °C/month")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# HEAT STRESS ANALYSIS
# ------------------------------------------------------------
print("---------------- HEAT STRESS ----------------")

heat_region_flag = heat_stress.any(dim="time")

heat_regions = int(heat_region_flag.sum())
heat_percent = heat_regions / heat_region_flag.size * 100

avg_heat_months = float(heat_stress.sum(dim="time").mean())

print(f"Regions experiencing heat stress (>35°C): {heat_regions} ({heat_percent:.2f}%)")
print(f"Average heat-stress duration: {avg_heat_months:.2f} months")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# COLD STRESS ANALYSIS
# ------------------------------------------------------------
print("---------------- COLD STRESS ----------------")

cold_region_flag = cold_stress.any(dim="time")

cold_regions = int(cold_region_flag.sum())
cold_percent = cold_regions / cold_region_flag.size * 100

avg_cold_months = float(cold_stress.sum(dim="time").mean())

print(f"Regions experiencing cold stress (<5°C): {cold_regions} ({cold_percent:.2f}%)")
print(f"Average frost-risk duration: {avg_cold_months:.2f} months")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# SOIL MOISTURE BASELINE
# ------------------------------------------------------------
print("---------------- SOIL MOISTURE ----------------")

soil_mean = float(swvl2_mean.mean()) * 100

print(f"Average deep soil moisture across Germany: {soil_mean:.1f}%")

moisture_min = float(swvl2_mean.min()) * 100
moisture_max = float(swvl2_mean.max()) * 100

print(f"Soil moisture range across regions: {moisture_min:.1f}% – {moisture_max:.1f}%")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# DROUGHT ANALYSIS
# ------------------------------------------------------------
print("---------------- DROUGHT RISK ----------------")

drought_region_flag = drought_flag_deep.any(dim="time")

drought_regions = int(drought_region_flag.sum())
drought_percent = drought_regions / drought_region_flag.size * 100

print(f"Regions experiencing drought (<0.15 soil moisture): {drought_regions} ({drought_percent:.2f}%)")

persistent_regions = int((drought_persistence >= 2).sum())
persistent_percent = persistent_regions / drought_region_flag.size * 100

print(f"Regions with consecutive drought months: {persistent_regions} ({persistent_percent:.2f}%)")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# SOIL TEMPERATURE STABILITY
# ------------------------------------------------------------
print("---------------- SOIL TEMPERATURE ----------------")

soil_temp_mean = float(soil_temp_l2.mean())

print(f"Average root-zone soil temperature: {soil_temp_mean:.2f} °C")

soil_temp_var = float(soil_temp_l2_std.mean())

print(f"Average soil temperature variability: {soil_temp_var:.2f} °C")

print("--------------------------------------------------\n")


# ------------------------------------------------------------
# CLIMATE STABILITY INDICATORS
# ------------------------------------------------------------
print("---------------- CLIMATE STABILITY ----------------")

temp_cv_mean = float(temp_cv.mean())
soil_cv_mean = float(swvl2_cv.mean())

print(f"Temperature variability (CV): {temp_cv_mean:.2f}")
print(f"Soil moisture variability (CV): {soil_cv_mean:.2f}")

print("--------------------------------------------------\n")

# =============================================================================================================
#                               PHASE 5B — REPORTING VISUALIZATIONS
# =============================================================================================================

import matplotlib.pyplot as plt
import os

# Create folder to store figures
os.makedirs("outputs/figures", exist_ok=True)


# ============================================================
# 1. SEASONAL TEMPERATURE CURVE (GERMANY AVERAGE)
# ============================================================

temp_spatial_mean = temp.mean(dim=("lat","lon"))

plt.figure(figsize=(8,5))

temp_spatial_mean.plot(marker="o")

plt.title("Seasonal Temperature Curve — Germany 2025")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/figures/seasonal_temperature_curve.png", dpi=300)

plt.show()


# ============================================================
# 2. GROWING DEGREE DAYS MAP
# ============================================================

plt.figure(figsize=(8,6))

gdd_annual.plot(cmap="YlOrRd")

plt.title("Growing Degree Days (Base 10°C)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

plt.savefig("outputs/figures/gdd_map.png", dpi=300)

plt.show()


# ============================================================
# 3. ROOT-ZONE SOIL MOISTURE MAP
# ============================================================

plt.figure(figsize=(8,6))

swvl2_mean.plot(cmap="YlGnBu")

plt.title("Average Root-Zone Soil Moisture")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

plt.savefig("outputs/figures/soil_moisture_map.png", dpi=300)

plt.show()


# ============================================================
# 4. DROUGHT RISK MAP
# ============================================================

plt.figure(figsize=(8,6))

drought_region_flag.plot(cmap="Reds")

plt.title("Regions Experiencing Drought Risk")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

plt.savefig("outputs/figures/drought_risk_map.png", dpi=300)

plt.show()

print("\nAll visualization figures saved to: outputs/figures/")

climate_features.to_parquet(
    "outputs/Climate_Features_2025.parquet",
    index=False
)
print(climate_features.columns)


# Group by location (lat/lon) and backfill only the gradients
climate_features = climate_features.sort_values(['lat', 'lon', 'time']) # Ensure chronological order first
climate_features[['Temp_Gradient', 'SWVL1_Gradient', 'SWVL2_Gradient']] = \
    climate_features.groupby(['lat', 'lon'])[['Temp_Gradient', 'SWVL1_Gradient', 'SWVL2_Gradient']].bfill()

# Verify the 5400 are gone
print(climate_features.isnull().sum())
print(climate_features.shape)
print("Saved: outputs/Climate_Features_2025.parquet")