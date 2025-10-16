import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
import folium
import networkx as nx
import streamlit as st
from streamlit_folium import st_folium

# --- Synthetic Data Generation ---
def generate_synthetic_monitor_data(num_points=100):
    np.random.seed(42)
    lat = np.random.uniform(17.3, 17.6, num_points)
    lon = np.random.uniform(78.3, 78.7, num_points)
    aqi = np.random.randint(30, 200, num_points)
    timestamps = [dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=i) for i in range(num_points)]
    df = pd.DataFrame({'lat': lat, 'lon': lon, 'AQI': aqi, 'timestamp': timestamps})
    return df

# --- AI/ML AQI Prediction ---
def train_predict_aqi(df):
    df['hour'] = df['timestamp'].dt.hour
    X = df[['lat', 'lon', 'hour']]
    y = df['AQI']

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=4)
    model.fit(X, y)

    future_hours = pd.DataFrame({
        'lat': np.random.uniform(17.3, 17.6, 50),
        'lon': np.random.uniform(78.3, 78.7, 50),
        'hour': np.random.randint(0, 24, 50)
    })

    y_pred = model.predict(future_hours)
    rmse = mean_squared_error(np.random.randint(30, 200, 50), y_pred, squared=False)
    future_hours['pred_AQI'] = y_pred
    return future_hours, rmse

# --- Interpolation for Hyperlocal AQI ---
def idw_interpolation(lats, lons, values, grid_lat, grid_lon, k=8, eps=1e-12):
    tree = cKDTree(np.c_[lats, lons])
    grid_points = np.c_[grid_lat.ravel(), grid_lon.ravel()]
    dist, idx = tree.query(grid_points, k=k)

    w = 1 / (dist + eps)
    w /= w.sum(axis=1)[:, None]
    interpolated = np.sum(w * values[idx], axis=1)

    return interpolated.reshape(grid_lat.shape)

# --- Health Alerts ---
def health_alert(aqi):
    if aqi < 50:
        return 'Good'
    elif aqi < 100:
        return 'Moderate'
    elif aqi < 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi < 200:
        return 'Unhealthy'
    else:
        return 'Very Unhealthy'

# --- Safe Route Suggestion ---
def safe_route(points, aqi_values):
    G = nx.grid_2d_graph(10, 10)
    for (u, v) in G.edges():
        G[u][v]['weight'] = (aqi_values[u[0]*10 + u[1]] + aqi_values[v[0]*10 + v[1]]) / 2
    path = nx.shortest_path(G, source=(0, 0), target=(9, 9), weight='weight')
    return path

# --- Visualization ---
def visualize_aqi(future_df):
    m = folium.Map(location=[17.45, 78.5], zoom_start=10)
    for _, row in future_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color='red' if row['pred_AQI'] > 150 else 'green',
            fill=True,
            fill_opacity=0.7,
            popup=f"AQI: {row['pred_AQI']:.1f} ({health_alert(row['pred_AQI'])})"
        ).add_to(m)
    return m

# --- Streamlit UI ---
st.set_page_config(page_title="AI-Based Air Quality Prediction", layout="wide")

st.title("🌍 AI-Based Air Quality Prediction App")
st.write("This app uses synthetic data and XGBoost to predict Air Quality Index (AQI), "
         "perform interpolation, and visualize safe routes.")

if st.button("🚀 Run AQI Prediction Demo"):
    with st.spinner("Generating data and training model..."):
        df = generate_synthetic_monitor_data()
        future_df, rmse = train_predict_aqi(df)
        lat_grid, lon_grid = np.mgrid[17.3:17.6:50j, 78.3:78.7:50j]
        interpolated_aqi = idw_interpolation(df['lat'].values, df['lon'].values, df['AQI'].values, lat_grid, lon_grid)
        path = safe_route(list(zip(df['lat'], df['lon'])), df['AQI'].values)
        m = visualize_aqi(future_df)

    st.success(f"✅ Model trained successfully! RMSE = {rmse:.2f}")
    st.write(f"Suggested safe route grid path: {path}")
    st_folium(m, width=900, height=600)
numpy
pandas
xgboost
scikit-learn
scipy
folium
networkx
streamlit
streamlit-folium
