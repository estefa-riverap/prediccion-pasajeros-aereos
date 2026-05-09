import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CARGAR ARCHIVOS
# ==========================================

modelo = joblib.load("modelo_gbr_FINAL.pkl")
scaler = joblib.load("scaler_FINAL.pkl")
columnas = joblib.load("columnas_FINAL.pkl")

# ==========================================
# TÍTULO
# ==========================================

st.title("Predicción de Pasajeros Aéreos")

st.write(
    "Predicción de pasajeros para rutas internacionales."
)

# ==========================================
# INPUTS
# ==========================================

anio = st.number_input("Año", 2020, 2035, 2024)

mes = st.selectbox(
    "Mes",
    [1,2,3,4,5,6,7,8,9,10,11,12]
)

vuelos = st.number_input(
    "Número de vuelos",
    min_value=1,
    value=120
)

horas = st.number_input(
    "Horas bloque",
    min_value=1.0,
    value=2500.0
)

sillas = st.number_input(
    "Sillas ofrecidas",
    min_value=1,
    value=25000
)

nombre_encoded = st.number_input(
    "Código aerolínea",
    min_value=0,
    value=1
)

ciudad_origen = st.selectbox(
    "Ciudad origen",
    [
        'BOGOTA DC',
        'RIONEGRO',
        'CARTAGENA DE INDIAS',
        'SANTIAGO DE CALI',
        'BARRANQUILLA'
    ]
)

ciudad_destino = st.selectbox(
    "Ciudad destino",
    [
        'MIAMI',
        'PANAMA',
        'NEW YORK',
        'ORLANDO',
        'PUNTA CANA',
        'MEXICO'
    ]
)

pais_destino = st.selectbox(
    "País destino",
    [
        'ESTADOS UNIDOS',
        'PANAMA',
        'MEXICO',
        'REPUBLICA DOMINICANA'
    ]
)

# ==========================================
# PREDICCIÓN
# ==========================================

if st.button("Predecir"):

    # Crear dataframe EXACTO del entrenamiento
    datos = pd.DataFrame(
        0,
        index=[0],
        columns=columnas
    )

    # Variables numéricas
    datos['Año'] = anio
    datos['Mes'] = mes
    datos['Número de Vuelos'] = vuelos
    datos['Horas Bloque'] = horas
    datos['Sillas Ofrecidas'] = sillas
    datos['Nombre_encoded'] = nombre_encoded

    # One Hot Encoding
    col_origen = f'Ciudad Origen_{ciudad_origen}'
    if col_origen in datos.columns:
        datos[col_origen] = 1

    col_destino = f'Ciudad Destino_{ciudad_destino}'
    if col_destino in datos.columns:
        datos[col_destino] = 1

    col_pais = f'Pais Destino_{pais_destino}'
    if col_pais in datos.columns:
        datos[col_pais] = 1

    # Escalar
    datos_scaled = scaler.transform(datos)

    # Predicción
    pred_log = modelo.predict(datos_scaled)[0]

    # Volver escala original
    pred_real = np.expm1(pred_log)

    # Resultado
    st.success(
        f"Cantidad estimada de pasajeros: {pred_real:,.0f}"
    )

    # Debug
    st.write("Predicción log:", pred_log)
    st.write("Predicción real:", pred_real)