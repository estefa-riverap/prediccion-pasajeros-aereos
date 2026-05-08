
import streamlit as st
import pandas as pd
import joblib

# ======================
# Cargar modelo y scaler
# ======================

modelo = joblib.load("modelo_gbr.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Título
# ======================

st.title("Predicción de Pasajeros Aéreos")

st.write(
    "Ingrese los datos del vuelo para estimar la cantidad de pasajeros."
)

# ======================
# Inputs
# ======================

anio = st.number_input(
    "Año",
    min_value=2020,
    max_value=2035,
    value=2025
)

mes = st.selectbox(
    "Mes",
    [1,2,3,4,5,6,7,8,9,10,11,12]
)

vuelos = st.number_input(
    "Número de vuelos",
    min_value=1,
    value=10
)

horas = st.number_input(
    "Horas bloque",
    min_value=0.0,
    value=100.0
)

sillas = st.number_input(
    "Sillas ofrecidas",
    min_value=1,
    value=1500
)

# ======================
# Predicción
# ======================

if st.button("Predecir"):

    datos = pd.DataFrame({
        "Año": [anio],
        "Mes": [mes],
        "Número de Vuelos": [vuelos],
        "Horas Bloque": [horas],
        "Sillas Ofrecidas": [sillas]
    })

    datos_scaled = scaler.transform(datos)

    pred = modelo.predict(datos_scaled)

    st.success(
        f"Cantidad estimada de pasajeros: {pred[0]:.0f}"
    )
