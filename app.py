import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CARGAR MODELO Y COLUMNAS
# ==========================================

modelo = joblib.load("modelo_gbr_FINAL.pkl")
columnas = joblib.load("columnas_FINAL.pkl")

# ==========================================
# TÍTULO
# ==========================================

st.title("Predicción de Pasajeros Aéreos")

st.write(
    "Aplicación para estimar la cantidad de pasajeros en rutas internacionales."
)

# ==========================================
# VARIABLES DE ENTRADA
# ==========================================

anio = st.number_input(
    "Año",
    min_value=2020,
    max_value=2035,
    value=2024
)

mes = st.selectbox(
    "Mes",
    [1,2,3,4,5,6,7,8,9,10,11,12]
)

numero_vuelos = st.number_input(
    "Número de vuelos",
    min_value=1,
    value=120
)

horas_bloque = st.number_input(
    "Horas bloque",
    min_value=1.0,
    value=2500.0
)

sillas_ofrecidas = st.number_input(
    "Sillas ofrecidas",
    min_value=1,
    value=25000
)

nombre_encoded = st.number_input(
    "Código aerolínea",
    min_value=0,
    value=1
)

# ==========================================
# CIUDAD ORIGEN
# ==========================================

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

# ==========================================
# CIUDAD DESTINO
# ==========================================

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

# ==========================================
# PAÍS DESTINO
# ==========================================

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
# BOTÓN PREDECIR
# ==========================================

if st.button("Predecir"):

    # Crear dataframe EXACTO del entrenamiento
    datos = pd.DataFrame(
        0,
        index=[0],
        columns=columnas
    )

    # ==========================================
    # VARIABLES NUMÉRICAS
    # ==========================================

    datos['Año'] = anio
    datos['Mes'] = mes
    datos['Número de Vuelos'] = numero_vuelos
    datos['Horas Bloque'] = horas_bloque
    datos['Sillas Ofrecidas'] = sillas_ofrecidas
    datos['Nombre_encoded'] = nombre_encoded

    # ==========================================
    # ONE HOT ENCODING
    # ==========================================

    col_origen = f'Ciudad Origen_{ciudad_origen}'

    if col_origen in datos.columns:
        datos[col_origen] = 1

    col_destino = f'Ciudad Destino_{ciudad_destino}'

    if col_destino in datos.columns:
        datos[col_destino] = 1

    col_pais = f'Pais Destino_{pais_destino}'

    if col_pais in datos.columns:
        datos[col_pais] = 1

    # ==========================================
    # PREDICCIÓN
    # ==========================================

    pred_log = modelo.predict(datos)[0]

    # ==========================================
    # VOLVER A ESCALA ORIGINAL
    # ==========================================

    pred_real = np.expm1(pred_log)

    # ==========================================
    # RESULTADO
    # ==========================================

    st.success(
        f"Cantidad estimada de pasajeros: {pred_real:,.0f}"
    )

    # Debug opcional
    st.write("Predicción log:", pred_log)
    st.write("Predicción real:", pred_real)