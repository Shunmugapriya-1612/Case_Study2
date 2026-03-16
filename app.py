import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------

st.set_page_config(
    page_title="Crop Suitability Advisor",
    layout="wide"
)

st.title("AI Crop Suitability Advisor")

st.write(
"""
Select a location and month to see the most suitable crops
based on climate, soil, and solar conditions.
"""
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    env = pd.read_parquet("outputs/AgroClimate_Feature_Matrix_2025.parquet")
    rank = pd.read_parquet("outputs/crop_recommendations.parquet")
    return env, rank

env_df, rank_df = load_data()
lat_values = sorted(env_df["lat"].unique())
lon_values = sorted(env_df["lon"].unique())
# ------------------------------------------------------------
# SESSION STATE FOR LAT/LON
# ------------------------------------------------------------

if "lat" not in st.session_state:
    st.session_state.lat = float(env_df["lat"].iloc[0])

if "lon" not in st.session_state:
    st.session_state.lon = float(env_df["lon"].iloc[0])

lat = st.session_state.lat
lon = st.session_state.lon

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

st.sidebar.header("Location Selection")

lat = st.sidebar.selectbox(
    "Latitude",
    lat_values,
    index=min(range(len(lat_values)), key=lambda i: abs(lat_values[i] - lat))
)

lon = st.sidebar.selectbox(
    "Longitude",
    lon_values,
    index=min(range(len(lon_values)), key=lambda i: abs(lon_values[i] - lon))
)
month = st.sidebar.selectbox(
    "Month",
    list(range(1,13))
)

# ------------------------------------------------------------
# LAYOUT (MAP + INFORMATION)
# ------------------------------------------------------------

map_col, info_col = st.columns([3,1])

# ------------------------------------------------------------
# MAP SECTION
# ------------------------------------------------------------

with map_col:

    st.subheader("Selected Location")

    st.markdown(f"""
📍 **Latitude:** {lat}  
📍 **Longitude:** {lon}
""")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=7,
        tiles="Esri.WorldImagery"
    )

    folium.Marker(
        [lat, lon],
        tooltip="Selected Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

    map_data = st_folium(
        m,
        width=None,
        height=520
    )

    # --------------------------------------------------------
    # CAPTURE MAP CLICK
    # --------------------------------------------------------

    if map_data["last_clicked"] is not None:

        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        nearest_lat = min(lat_values, key=lambda x: abs(x - clicked_lat))
        nearest_lon = min(lon_values, key=lambda x: abs(x - clicked_lon))

        st.session_state.lat = nearest_lat
        st.session_state.lon = nearest_lon

        st.rerun()

        
    
# ------------------------------------------------------------
# INFORMATION PANEL
# ------------------------------------------------------------

with info_col:

    # --------------------------------------------------------
    # ENVIRONMENT CONDITIONS
    # --------------------------------------------------------

    st.subheader("Environmental Conditions")

    env_selected = env_df[
        (env_df["lat"] == lat) &
        (env_df["lon"] == lon) &
        (env_df["time"].dt.month == month)
    ]

    if len(env_selected) > 0:

        env = env_selected.iloc[0]

        st.markdown(f"""
**Air Temperature (°C)**  
{env['Temp']:.2f}

**Surface Soil Temperature (°C)**  
{env['SoilTemp_L1']:.2f}

**Root Zone Soil Temperature (°C)**  
{env['SoilTemp_L2']:.2f}

**Surface Soil Moisture (m³/m³)**  
{env['SWVL1']:.2f}

**Root Zone Soil Moisture (m³/m³)**  
{env['SWVL2']:.2f}

**Daily Light Integral (mol/m²/day)**  
{env['DLI_value']:.2f}

**Daylight Duration (hours)**  
{env['Daylight_hours']:.2f}
""")

    else:

        st.warning("No environmental data found.")

    # --------------------------------------------------------
    # CROP RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("Top Suitable Crops")

    crop_results = rank_df[
        (rank_df["lat"] == lat) &
        (rank_df["lon"] == lon) &
        (rank_df["month"] == month)
    ]

    if len(crop_results) > 0:

        st.success(
            crop_results["Top_Crops"].values[0]
        )

    else:

        st.warning("No crop recommendations found.")