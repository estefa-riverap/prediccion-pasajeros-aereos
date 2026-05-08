import streamlit as st
import pandas as pd
import joblib

# ======================
# Cargar archivos
# ======================

modelo = joblib.load("modelo_gbr.pkl")
scaler = joblib.load("scaler.pkl")
encoder_nombre = joblib.load("encoder_nombre.pkl")

# ======================
# Título
# ======================

st.title("Predicción de Pasajeros Aéreos")

st.write(
    "Ingrese la información del vuelo."
)

# ======================
# Inputs
# ======================

anio = st.number_input("Año", 2020, 2035, 2025)

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

# Aerolínea
nombre = st.selectbox(
    "Aerolínea",
    list(encoder_nombre.classes_)
)

# Ciudad origen
ciudad_origen = st.selectbox(
    "Ciudad origen",
    [
        'ARMENIA',
        'BARRANCA DE UPIA',
        'BARRANQUILLA',
        'BOGOTA DC',
        'BUCARAMANGA',
        'CARTAGENA DE INDIAS',
        'PEREIRA',
        'RIONEGRO',
        'RIONEGRO ANTIOQUIA',
        'RIONEGRO SANTANDER',
        'SAN ANDRES',
        'SAN ANDRES ISLA',
        'SAN JOSE DE CUCUTA',
        'SANTA MARTA',
        'SANTIAGO DE CALI'
    ]
)

# País destino
pais_destino = st.selectbox(
    "País destino",
    [
        'ANTILLAS HOLANDESAS',
        'ESTADOS UNIDOS',
        'MEXICO',
        'PANAMA',
        'REPUBLICA DOMINICANA'
    ]
)

# ======================
# Predicción
# ======================

if st.button("Predecir"):

    columnas = [
        'Año', 'Mes', 'Número de Vuelos',
        'Horas Bloque', 'Sillas Ofrecidas',
        'Nombre_encoded',
        'Ciudad Origen_ARMENIA',
        'Ciudad Origen_BARRANCA DE UPIA',
        'Ciudad Origen_BARRANQUILLA',
        'Ciudad Origen_BOGOTA DC',
        'Ciudad Origen_BUCARAMANGA',
        'Ciudad Origen_CARTAGENA DE INDIAS',
        'Ciudad Origen_PEREIRA',
        'Ciudad Origen_RIONEGRO',
        'Ciudad Origen_RIONEGRO ANTIOQUIA',
        'Ciudad Origen_RIONEGRO SANTANDER',
        'Ciudad Origen_SAN ANDRES',
        'Ciudad Origen_SAN ANDRES ISLA',
        'Ciudad Origen_SAN JOSE DE CUCUTA',
        'Ciudad Origen_SANTA MARTA',
        'Ciudad Origen_SANTIAGO DE CALI',
        'Pais Destino_ANTILLAS HOLANDESAS',
        'Pais Destino_ESTADOS UNIDOS',
        'Pais Destino_MEXICO',
        'Pais Destino_PANAMA',
        'Pais Destino_REPUBLICA DOMINICANA'
    ]

    # Crear dataframe vacío
    datos = pd.DataFrame(0, index=[0], columns=columnas)

    # Variables numéricas
    datos['Año'] = anio
    datos['Mes'] = mes
    datos['Número de Vuelos'] = vuelos
    datos['Horas Bloque'] = horas
    datos['Sillas Ofrecidas'] = sillas

    # Label Encoding
    datos['Nombre_encoded'] = encoder_nombre.transform([nombre])[0]

    # One Hot Encoding origen
    col_origen = f'Ciudad Origen_{ciudad_origen}'

    if col_origen in datos.columns:
        datos[col_origen] = 1

    # One Hot Encoding país
    col_pais = f'Pais Destino_{pais_destino}'

    if col_pais in datos.columns:
        datos[col_pais] = 1

    # Escalar
    datos_scaled = scaler.transform(datos)

    # Predicción
    pred = modelo.predict(datos_scaled)

    # Resultado
    st.success(
        f"Pasajeros estimados: {pred[0]:.0f}"
    )