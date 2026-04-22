# ============================
# STREAMLIT CLOUD APP (NO TABPY)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="LOS Prediction Dashboard", layout="wide")

st.title("🏥 Length of Stay Prediction Dashboard")
st.write("This app predicts patient Length of Stay using a Random Forest model.")

# 🔹 Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df)

    # 🔹 Clean data
    df = df.dropna()

    # 🔹 Sidebar filters (adds interactivity)
    st.sidebar.header("🔎 Filters")

    if 'DeptID' in df.columns:
        selected_dept = st.sidebar.selectbox("Select Department", df['DeptID'].unique())
        df = df[df['DeptID'] == selected_dept]

    # 🔹 Feature Engineering
    X = pd.get_dummies(df[['AdmissionType', 'DeptID', 'Age', 'Primary Condition']])
    y = df['ALOS']

    # 🔹 Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 🔹 Predictions
    df['Predicted_ALOS'] = model.predict(X)

    st.subheader("📈 Predictions")
    st.dataframe(df[['ALOS', 'Predicted_ALOS']])

    # 🔹 Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#262626')
    ax.set_facecolor('#262626')

    # Scatter
    ax.scatter(df['ALOS'], df['Predicted_ALOS'],
               color='#D4AF37', alpha=0.5, label='Patient Records')

    # Perfect line
    ax.plot([df['ALOS'].min(), df['ALOS'].max()],
            [df['ALOS'].min(), df['ALOS'].max()],
            linestyle='--', color='white', label='Perfect Prediction')

    # 🔹 Trendline
    z = np.polyfit(df['ALOS'], df['Predicted_ALOS'], 1)
    p = np.poly1d(z)

    ax.plot(df['ALOS'], p(df['ALOS']),
            color='#00FFFF', linewidth=2, label='Trendline')

    # Labels
    ax.set_xlabel('Actual Length of Stay (Days)', color='white')
    ax.set_ylabel('Predicted Length of Stay (Days)', color='white')
    ax.set_title('Actual vs Predicted LOS', color='white')
    ax.tick_params(colors='white')
    ax.legend()

    st.pyplot(fig)

    # 🔹 Equation
    st.write(f"📐 Trendline Equation: y = {z[0]:.2f}x + {z[1]:.2f}")

else:
    st.info("⬆️ Upload a dataset to begin")