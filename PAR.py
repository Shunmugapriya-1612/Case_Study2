import xarray as xr
import numpy as np
import glob

# ------------------------------------------------------------
# Load all 12 monthly files and unwrapping .nc files
# ------------------------------------------------------------
files = sorted(glob.glob("PARmm2025*.nc"))
print("Number of files:", len(files))

datasets = []
for f in files:
    ds_single = xr.open_dataset(
        f,
        engine="netcdf4",   
        mask_and_scale=True,
        decode_cf=True
    )

    # Extracting only "PAR" column values, as the other columns(PARC,PAR_dms and record_status) are irrevalant
    datasets.append(ds_single[["PAR"]])

# Concatenate months along time(-means(which Month)) as coordinate, along with lat and lon
# Concatenate months along time
ds = xr.concat(datasets, dim="time")

# ------------------------------------------------------------
# Restrict Dataset to Germany Region
# ------------------------------------------------------------

GERMANY_LAT_MIN = 47
GERMANY_LAT_MAX = 55
GERMANY_LON_MIN = 5
GERMANY_LON_MAX = 15

ds = ds.sel(
    lat=slice(GERMANY_LAT_MIN, GERMANY_LAT_MAX),
    lon=slice(GERMANY_LON_MIN, GERMANY_LON_MAX)
)

# ============================================================
# PHASE 1.5 — Spatial Harmonization (0.05° → 0.1°)
# ============================================================

ds = ds.coarsen(lat=2, lon=2, boundary="trim").mean()

print("Solar dataset downscaled to match climate resolution")
print("New dimensions after coarsening:", dict(ds.sizes))
      
print("Dataset restricted to Germany")
print("New dimensions:", dict(ds.sizes))
#the dataset ds is of format "xarray" - which shows the data with coordinates(lat,lan) - PAR data in particular (lon,lat).
print("---------------------------")
print("Coordinate names:", list(ds.coords))
print("Variable names:", list(ds.data_vars))
print("---------------------------")
#-------------------------------------------------------------------------------
#                          Feature Engineering 
#-------------------------------------------------------------------------------
#for learning seasonal trends three data is required.
#1. Lat - fixed location
#2. Declination degree(angle - rotation across the sun) - where the sun is over(equator - sunlight hours will be higher /tropic of cancer /tropic of capricon - both tropics change depending on seasons)
#3. omega(angle - rotation, Earth's own rotation - day and night) -  how long the sun stayed on the lat before sunset

#Satellite data is stored in degrees(lat,lon) but the below trignometric formula(for solar declination) require data in radians 
#hence the data in degree is converted to randian
lat_rad = np.deg2rad(ds["lat"])

# the month's data is converted to day's data - (Jan - 1,feb - 32, mar - 60) for solar declination
day_of_year = ds["time"].dt.dayofyear

# Solar declination (FAO-56 formulation) - to find seasonal migration(whether the sun is over equator/tropic of cancer/tropic of capricon)/trends of sunlight
decl = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)

#Declination angle and Latitude is converted to 2d because this is needed for next dot product and tells "how long the sunlight is available at the location"
lat_2d, decl_2d = xr.broadcast(lat_rad, decl)


# Argument inside arccos - says how long the lat(earth's position) is under the sun
arg = -np.tan(lat_2d) * np.tan(decl_2d)

#-------------------------------
#        Normalization 
#-------------------------------

#Normalizing the angle values, why - for eg - -12 degree - says there was no sunrise(no daylight) and +12 means there was no sunset(24 hour sunlight)
arg = xr.where(arg > 1, 1, arg)
arg = xr.where(arg < -1, -1, arg)

# Angle of Earth's rotation
omega = np.arccos(arg)

# Hours of sunlight per day, one day, 24 hours = 2.pi radian hence 1 angle radian rotation takes 12/pi hours
daylength_hours = (24 / np.pi) * omega

# Daylength hours is calculated based on lat, hence broadcasting the same hours to longtitude as well to match with ds dataset - PAR,lat,lon
daylength_hours = daylength_hours.broadcast_like(ds)

#          lon →
#       ┌───────────────┐
#  lat ↓│               │
#       │     PAR       │  ← values
#       │               │
#       └───────────────┘
#            ↑
#           time (stacked months)


# ------------------------------------------------------------
# 3️⃣ Convert Monthly Mean PAR → Daily Light Integral (DLI)
# ------------------------------------------------------------
# ds - mean PAR per month, daylength_hours - hours of sunlight per day
# multiplied with 0.0036 because par unit is mol photons per m2 per second, and for daylength_hours - unit(Converting to seconds) - 1 hour =  3600 seconds 
# 1 mol=10^6 µmol, 3600/10^6=0.0036

DLI = ds["PAR"] * daylength_hours * 0.0036

#Naming DLI as a variable value on 3d(time, lat, lon) Plan
DLI.name = "DLI"

# ------------------------------------------------------------
# 4️⃣ Create Dataset with DLI
# ------------------------------------------------------------
#Create a dataset where the variable is only DLI
ds_dli = xr.Dataset({"DLI": DLI})

jan_mean = float(ds_dli["DLI"].isel(time=0).mean().values)
jul_mean = float(ds_dli["DLI"].isel(time=6).mean().values)

print("\n---------------- DLI SUMMARY ----------------")

print(f"Average DLI across Germany in January 2025: {jan_mean:.2f} mol/m²/day")

print(f"Average DLI across Germany in July 2025: {jul_mean:.2f} mol/m²/day")

print("\nDataset Dimensions:")
print(f"Time steps (months): {ds_dli.sizes['time']}")
print(f"Latitude grid points: {ds_dli.sizes['lat']}")
print(f"Longitude grid points: {ds_dli.sizes['lon']}")

print("------------------------------------------------\n")

# ------------------------------------------------------------
# 🔎 Inspect ONLY valid satellite pixels (ignore masked areas)
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4️⃣ Remove empty satellite regions (crop to valid footprint)
# ------------------------------------------------------------

# Find pixels that have data in at least one month
# True  = pixel observed at least once in 2025
# False = always NaN (outside satellite view)

valid_mask = ds["PAR"].notnull().any(dim="time")

# Determine bounding box of valid data
# For each latitude, is there at least one valid pixel across longitudes?
valid_lat = valid_mask.any(dim="lon")

# For each longitude , is there at least one valid pixel across latitudes?
valid_lon = valid_mask.any(dim="lat")

# above Valid_lat and Valid_Lon will be list of boolean values, below command converts boolean to the rows number  only 'Trues's are considered and ignore False's.
lat_indices = np.where(valid_lat)[0]
lon_indices = np.where(valid_lon)[0]

#identify border of the rectagular grid with relastic data, and identifies how much empty data needs to be cropped.
lat_min, lat_max = lat_indices[0], lat_indices[-1]
lon_min, lon_max = lon_indices[0], lon_indices[-1]

# Crop dataset to only observed region
ds_dli_cropped = ds_dli.isel(
    lat=slice(lat_min, lat_max + 1),
    lon=slice(lon_min, lon_max + 1)
)


print("\n---------------- DATA CROPPING ----------------")
print(f"Latitude range: {float(ds_dli_cropped['lat'].min().values):.2f}° "
      f"to {float(ds_dli_cropped['lat'].max().values):.2f}°")

print(f"Longitude range: {float(ds_dli_cropped['lon'].min().values):.2f}° "
      f"to {float(ds_dli_cropped['lon'].max().values):.2f}°")
print(f"Final dataset shape after cropping: {dict(ds_dli_cropped.sizes)}")
print("------------------------------------------------\n")
# ------------------------------------------------------------
# 5️⃣ Save Cleaned Dataset (no huge NaN borders)
# ------------------------------------------------------------

encoding = {"DLI": {"zlib": True, "complevel": 4}}  # compressed output

ds_dli_cropped.to_netcdf("DLI_2025_monthly.nc", encoding=encoding)

print("Saved: DLI_2025_monthly.nc")
# ------------------------------------------------------------
# PHASE 2 — Convert raw light into biologically meaningful indicators
# ------------------------------------------------------------
# ------------------------------------------------------------
# 2A — Photoperiod Characterization ("How long daylight lasts")
# ------------------------------------------------------------
# Daylength depends only on latitude (earth revolve through each lat across the sun)→ remove longitude duplication
# Dimension Reduction by averaging
photoperiod_lat = daylength_hours.mean(dim="lon")

# Annual statistics
# Avg daylight at each latitude
photo_mean = photoperiod_lat.mean(dim="time")

# Longest day at each latitude
photo_max  = photoperiod_lat.max(dim="time")

# Shortest day at each latitude
photo_min  = photoperiod_lat.min(dim="time")

# Month when longest day occurs (defines growing season timing)
photo_peak_month = photoperiod_lat.idxmax(dim="time")
print("\n---------------- PHOTOPERIOD SUMMARY ----------------")

print("Average daylight duration across region (hours):",
      f"{float(photo_mean.mean().values):.2f}")

print("Longest daylight observed anywhere (hours):",
      f"{float(photo_max.max().values):.2f}")

print("Shortest daylight observed anywhere (hours):",
      f"{float(photo_min.min().values):.2f}")

print("Month with longest days:",
      str(photo_peak_month.values[0])[:7])  # show YYYY-MM only

print("-----------------------------------------------------\n")
# ------------------------------------------------------------
# 2B — Seasonal DLI Behaviour
# ------------------------------------------------------------

# Mean DLI per month (spatial average)
# Dimensional reduction: (time × lat × lon) → (time)
# Each value now represents the regional average Daily Light Integral for that month
dli_monthly_mean = ds_dli_cropped["DLI"].mean(dim=("lat", "lon"))

# Identify which month receives the highest total photosynthetic light
dli_peak_month = dli_monthly_mean.idxmax(dim="time")

# Identify which month receives the lowest total photosynthetic light
dli_min_month  = dli_monthly_mean.idxmin(dim="time")

# Seasonal amplitude:
# Difference between brightest and darkest month
annual_light_range = dli_monthly_mean.max() - dli_monthly_mean.min()

print("---------------- RADIATION SEASONALITY ----------------")

print("Month with highest available light:", str(dli_peak_month.values)[:7])
print("Month with lowest available light :", str(dli_min_month.values)[:7])

print("Annual light range (difference between brightest and darkest months):",
      f"{float(annual_light_range.values):.2f} mol/m²/day")

print("------------------------------------------------------\n")

# ------------------------------------------------------------
# 2C — Stability / Variability Metrics
# ------------------------------------------------------------

# Measures how strongly the total photosynthetic light (DLI) changes through the year
# at each location (lat, lon).
# High value → strong growing season vs dormant season contrast.
# Low value  → light availability is stable year-round.
dli_variability = ds_dli_cropped["DLI"].std(dim="time")

# Relative variability of light availability (Coefficient of Variation).
# Normalizes fluctuations by the average DLI so regions with different brightness
# levels can be compared fairly.
#
# Low CV  → stable solar climate (predictable for crops)
# High CV → unstable light regime (higher agricultural risk)
dli_cv = dli_variability / ds_dli_cropped["DLI"].mean(dim="time")

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Map 1 — Seasonal Variability (Absolute Change)
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
dli_variability.plot()
plt.title("Seasonal Change in Daily Light Integral\n(Standard Deviation across 12 months)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# ------------------------------------------------------------
# Map 2 — Relative Variability (Radiation Reliability)
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
dli_cv.plot(cbar_kwargs={"label": "Coefficient of Variation of DLI"})
plt.title("Radiation Reliability Map\n(Coefficient of Variation)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# ------------------------------------------------------------
# 2D — Light Sufficiency Metrics (Crop Usability Indicators)
# ------------------------------------------------------------

# Define agronomic thresholds (mol/m²/day)
LOW_LIGHT  = 12
MID_LIGHT  = 20
HIGH_LIGHT = 30

# Count how many months meet each threshold
months_low  = (ds_dli_cropped["DLI"] >= LOW_LIGHT).sum(dim="time")
months_mid  = (ds_dli_cropped["DLI"] >= MID_LIGHT).sum(dim="time")
months_high = (ds_dli_cropped["DLI"] >= HIGH_LIGHT).sum(dim="time")

# Compute national average number of suitable months
avg_low_months  = float(months_low.mean().values)
avg_mid_months  = float(months_mid.mean().values)
avg_high_months = float(months_high.mean().values)

print("\n---------------- LIGHT SUITABILITY SUMMARY ----------------")

print(f"Low-light crops are suitable for approximately '{round(avg_low_months)}' months per year on average.")

print(f"Moderate-light crops are suitable for approximately '{round(avg_mid_months)}' months per year on average.")

print(f"High-light crops are suitable for approximately '{round(avg_high_months)}' months per year on average.")

print("-----------------------------------------------------------\n")
# ------------------------------------------------------------
# PHASE 3 — Analyze how light changes month-to-month
# -----------------------------------------------------------
# ------------------------------------------------------------
# 3A — Seasonal Curve Features
# ------------------------------------------------------------

# How much solar energy changed from one month to the next, same lat and lon across consecutive months
dli_gradient = dli_monthly_mean.diff(dim="time")

# When does the growing season “switch on” most aggressively?
# High value → sudden growing season (continental climate)
# Low value → gradual transition (maritime climate)
max_growth_rate = dli_gradient.max()

# When does the season collapse fastest?
max_decline_rate = dli_gradient.min()

# Duration of high-light season (months above annual mean)
# No. Of Months when radiation is agriculturally productive
above_mean = dli_monthly_mean > dli_monthly_mean.mean()
season_length = above_mean.sum()

#At some point in the year, plants suddenly start receiving 0.78 more moles of photons per day compared to the previous month.
#This is the spring ramp-up of photosynthetic energy.
print("Fastest seasonal increase :", f"{float(max_growth_rate.values):.2f} mol/m²/day per month")

#After peak season, light availability drops by about 0.57 mol/m²/day each month at the fastest decline.

#This is the autumn shutdown speed.,
print("Fastest seasonal decline  :", f"{float(max_decline_rate.values):.2f} mol/m²/day per month")
print("Length of main light season:", int(season_length.values), "months\n")

# ------------------------------------------------------------
# 3B — Peak Structure
# ------------------------------------------------------------
# brightest month of the year
peak_value = dli_monthly_mean.max()

# No. Of Months where light is still agriculturally meaningful.
# Below ~50% → insufficient radiation for many crops
# Above ~50% → productive photosynthesis possible
half_peak_threshold = peak_value * 0.5

# Months sustaining at least 50% of peak radiation
broad_peak_duration = (dli_monthly_mean >= half_peak_threshold).sum()

print("Maximum of the monthly averaged DLI across all Germany grid cells : ", f"{float(peak_value.values):.2f} mol/m²/day")
print("No. Of Months sustaining ≥50% of peak light:", int(broad_peak_duration.values), "months\n")

# ------------------------------------------------------------
# PHASE 4 — Understand how light behaves differently at each location
# ------------------------------------------------------------
# ------------------------------------------------------------
# 4A — Mean Annual DLI
# ------------------------------------------------------------

# For each latitude & longitude:
# Take the average of DLI across all 12 months.
# This tells us how bright that location is on average throughout the year.
mean_annual_dli = ds_dli_cropped["DLI"].mean(dim="time")
# ------------------------------------------------------------
# 4B — Annual Amplitude
# ------------------------------------------------------------

# For each location:
# Find the brightest month and the darkest month.
# Subtract them.
# This tells us how strong the seasonal difference is at that location.
annual_amplitude_map = (
    ds_dli_cropped["DLI"].max(dim="time")
    - ds_dli_cropped["DLI"].min(dim="time")
)

# ------------------------------------------------------------
# 4C — Peak Month Timing
# ------------------------------------------------------------

# For each location:
# Identify which month has the highest DLI.
# This tells us when the strongest sunlight happens at that place.
peak_month_map = ds_dli_cropped["DLI"].idxmax(dim="time")

# ------------------------------------------------------------
# 4D — Stability (Coefficient of Variation)
# ------------------------------------------------------------

# We already calculated variability earlier (standard deviation over time).
# Now we use the relative version (CV).
# This tells us how stable or unstable the light pattern is at each location.
# Low value  → light is steady through the year.
# High value → light fluctuates strongly between seasons.
stability_map = dli_cv

# ------------------------------------------------------------
# 4E — Solar Regime Classification
# ------------------------------------------------------------

# We now classify each location into a simple category based on:
# 1. How bright it is (mean annual DLI)
# 2. How stable it is (CV)

# Classification meaning:
# 1 → Bright and stable (good year-round light)
# 2 → Bright but seasonal (strong summer peak)
# 3 → Moderate and stable
# 4 → Low light and unstable

solar_regime = xr.where(
    (mean_annual_dli > 20) & (stability_map < 0.20),
    1,
    xr.where(
        (mean_annual_dli > 20) & (stability_map >= 0.20),
        2,
        xr.where(
            (mean_annual_dli <= 12) & (stability_map < 0.30),
            3,
            4
        )
    )
)
print(
    "Mean Annual DLI range:",
    f"{float(mean_annual_dli.min()):.2f}",
    "to",
    f"{float(mean_annual_dli.max()):.2f}",
    "mol/m²/day"
)

print(
    "Stability (CV) range:",
    f"{float(stability_map.min()):.2f}",
    "to",
    f"{float(stability_map.max()):.2f}"
)

# ------------------------------------------------------------
# 4F — Visualize Results
# ------------------------------------------------------------

import matplotlib.pyplot as plt

# Map 1: Average yearly light level
plt.figure(figsize=(7,5))
mean_annual_dli.plot()
plt.title("Mean Annual DLI (Average Light Per Location)")
plt.show()

# Map 2: Seasonal strength
plt.figure(figsize=(7,5))
annual_amplitude_map.plot()
plt.title("Annual Light Change (Seasonal Strength)")
plt.show()


# Map 3: Solar regime categories
import matplotlib.colors as mcolors

plt.figure(figsize=(7,5))

cmap = mcolors.ListedColormap(["green", "orange", "blue", "red"])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

img = plt.imshow(solar_regime, cmap=cmap, norm=norm)

cbar = plt.colorbar(img, ticks=[1,2,3,4])
cbar.ax.set_yticklabels([
    "Bright_Stable",
    "Bright_Seasonal",
    "Moderate_Stable",
    "Low_Unstable"
])

plt.title("Solar Climate Categories")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# ------------------------------------------------------------
# PHASE 5 — Build Full Spatial Solar Feature Matrix
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5A — Base Monthly DLI (lat × lon × time)
# ------------------------------------------------------------
# Convert 3D xarray (time × lat × lon) into a table.
#
# Output example:
# time | lat | lon | DLI_value
#
# DLI_value represents:
# Total biologically usable light(PAR) energy (mol/m²/day) for that location in that month.

solar_features = ds_dli_cropped["DLI"].to_dataframe().reset_index()
solar_features.rename(columns={"DLI": "DLI_value"}, inplace=True)
solar_features = solar_features.reset_index(drop=True)

# ------------------------------------------------------------
# 5B — Daylight Hours (Monthly, Spatial)
# ------------------------------------------------------------
# Add number of daylight hours per location per month.
#
# How long the sun stays above the horizon.
# Important:
# Long daylight does NOT always mean high DLI
# (clouds can also reduce usable radiation).
daylength_hours.name = "Daylight_hours"

daylight_df = daylength_hours.to_dataframe().reset_index()
daylight_df.rename(columns={0: "Daylight_hours"}, inplace=True)

solar_features = solar_features.merge(
    daylight_df,
    on=["time", "lat", "lon"],
    how="left"
)


# ------------------------------------------------------------
# 5C — Month-to-Month Gradient
# ------------------------------------------------------------
# Calculate how much light changed compared to the previous month.
# DLI_gradient
#
# Interpretation:
# Positive value → light increasing (spring ramp-up)
# Negative value → light decreasing (autumn decline)
#
# Large magnitude → rapid seasonal transition
# Small magnitude → smooth seasonal change

dli_gradient_spatial = ds_dli_cropped["DLI"].diff(dim="time")

gradient_df = dli_gradient_spatial.to_dataframe().reset_index()
gradient_df.rename(columns={"DLI": "DLI_gradient"}, inplace=True)

solar_features = solar_features.merge(
    gradient_df,
    on=["time", "lat", "lon"],
    how="left"
)


# ------------------------------------------------------------
# 5D — Above Annual Mean (Reuse Phase 4 Mean)
# ------------------------------------------------------------
# Compare each month’s DLI with that location’s annual average.
# Above_Annual_Mean
#
# 1 → Month is brighter than that location’s yearly average
# 0 → Month is darker than average
#
# Interpretation:
# Identifies active growing season vs low-light season.
above_mean_spatial = ds_dli_cropped["DLI"] > mean_annual_dli

above_df = above_mean_spatial.to_dataframe().reset_index()
above_df.rename(columns={"DLI": "Above_Annual_Mean"}, inplace=True)

above_df["Above_Annual_Mean"] = above_df["Above_Annual_Mean"].astype(int)

solar_features = solar_features.merge(
    above_df,
    on=["time", "lat", "lon"],
    how="left"
)


# ------------------------------------------------------------
# 5E — Peak Month Flag (Reuse Phase 4 Peak Map)
# ------------------------------------------------------------
# Identify the brightest month of the year for each location.
# Is_Peak_Month
#
# 1 → This month is the peak radiation month
# 0 → Not peak
#
# Interpretation:
# Helps identify the strongest productivity window.
# Convert peak month map to dataframe with proper name
peak_df = peak_month_map.to_dataframe(name="Peak_Month").reset_index()

solar_features = solar_features.merge(
    peak_df,
    on=["lat", "lon"],
    how="left"
)

solar_features["Is_Peak_Month"] = (
    solar_features["time"] == solar_features["Peak_Month"]
).astype(int)

solar_features.drop(columns=["Peak_Month"], inplace=True)


# ------------------------------------------------------------
# 5F — Light Suitability Flags
# ------------------------------------------------------------
# Direct crop usability signal for each location-month.

# LOW_LIGHT  → shade-tolerant crops
# MID_LIGHT  → moderate-light crops
# HIGH_LIGHT → sun-demanding crops

LOW_LIGHT  = 12
MID_LIGHT  = 20
HIGH_LIGHT = 30

# Create binary indicators:
# 1 → Month meets light requirement
# 0 → Not suitable

solar_features["Suitable_Low"]  = (solar_features["DLI_value"] >= LOW_LIGHT).astype(int)
solar_features["Suitable_Mid"]  = (solar_features["DLI_value"] >= MID_LIGHT).astype(int)
solar_features["Suitable_High"] = (solar_features["DLI_value"] >= HIGH_LIGHT).astype(int)


# ------------------------------------------------------------
# 5G — Optimal Light Hours
# ------------------------------------------------------------
# Estimate how many daylight hours are actually useful
# for productive crop growth.
#
# If MID light requirement is met:
# → assume full daylight hours are productive
#
# If not:
# → productive hours = 0
solar_features["Optimal_hours"] = (
    solar_features["Daylight_hours"] *
    solar_features["Suitable_Mid"]
)
# ------------------------------------------------------------
# 5H — Merge Static Spatial Features
# ------------------------------------------------------------
# Mean Annual DLI
# Average yearly brightness of that location.
# High → generally bright region
# Low → generally darker region
# Mean Annual DLI
mean_df = mean_annual_dli.to_dataframe().reset_index()
mean_df.rename(columns={"DLI": "Mean_Annual_DLI"}, inplace=True)

solar_features = solar_features.merge(
    mean_df,
    on=["lat", "lon"],
    how="left"
)

# Annual Amplitude
# Difference between brightest and darkest month.
# High → strong seasons
# Low → uniform light through year

amp_df = annual_amplitude_map.to_dataframe().reset_index()
amp_df.rename(columns={"DLI": "Annual_Amplitude"}, inplace=True)

solar_features = solar_features.merge(
    amp_df,
    on=["lat", "lon"],
    how="left"
)

# Radiation Stability (Coefficient of Variation)
# Measures how stable light is throughout the year.
# Low value → predictable climate
# High value → strong seasonal contrast

stability_df = stability_map.to_dataframe().reset_index()
stability_df.rename(columns={"DLI": "Radiation_Stability"}, inplace=True)

solar_features = solar_features.merge(
    stability_df,
    on=["lat", "lon"],
    how="left"
)
# Solar Regime Classification
# Climate-type label based on brightness and stability.
#
# Categories:
# Bright_Stable
# Bright_Seasonal
# Moderate_Stable
# Low_Unstable

regime_named = xr.where(
    solar_regime == 1, "Bright_Stable",
    xr.where(
        solar_regime == 2, "Bright_Seasonal",
        xr.where(
            solar_regime == 3, "Moderate_Stable",
            "Low_Unstable"
        )
    )
)

regime_df = regime_named.to_dataframe().reset_index()
regime_df.rename(columns={"DLI": "Solar_Regime"}, inplace=True)

solar_features = solar_features.merge(
    regime_df,
    on=["lat", "lon"],
    how="left"
)


# ------------------------------------------------------------
# Final Check
# ------------------------------------------------------------

print("\nFinal Solar Feature Matrix Preview:\n")
print(solar_features.head())
print(solar_features.columns)
print("\nsolar_feature_df - Shape:", solar_features.shape)

print(solar_features.info())
# Save solar feature matrix
solar_features.to_parquet(
    "outputs/Solar_Features_2025.parquet",
    index=False
)

print("Saved: outputs/Solar_Features_2025.parquet")