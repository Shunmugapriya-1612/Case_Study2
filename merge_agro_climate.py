import pandas as pd

solar_df = pd.read_parquet("outputs/Solar_Features_2025.parquet")
climate_df = pd.read_parquet("outputs/Climate_Features_2025.parquet")

# 1. Round coordinates to prevent "float jitter" issues during merge
for df in [solar_df, climate_df]:
    df['lat'] = df['lat'].astype(float).round(1)
    df['lon'] = df['lon'].astype(float).round(1)

print("Solar dataset:", solar_df.shape)
print("Climate dataset:", climate_df.shape)

solar_df = solar_df[solar_df['time'] >= '2025-01-31']
climate_df['time'] = climate_df['time'] + pd.Timedelta(days=1)

# 2. Merge using 'inner' to keep only land pixels with full data
agro_climate_df = solar_df.merge(
    climate_df,
    on=["time", "lat", "lon"],
    how="inner"  # Changed from 'left' to avoid new NaNs from water pixels
)
agro_climate_df = agro_climate_df.sort_values(['lat', 'lon', 'time'])

# 2. Group by location and backfill the gradients
gradient_cols = ['Temp_Gradient', 'SWVL1_Gradient', 'SWVL2_Gradient']
agro_climate_df[gradient_cols] = (
    agro_climate_df.groupby(['lat', 'lon'])[gradient_cols]
    .bfill()
)

print("Merged dataset shape:", agro_climate_df.shape)
print("Total Nulls in Merged Data:\n", agro_climate_df.isna().sum().sum())





# 3. Save
agro_climate_df.to_parquet(
    "outputs/AgroClimate_Feature_Matrix_2025.parquet",
    index=False
)

print("Saved: AgroClimate_Feature_Matrix_2025.parquet")
