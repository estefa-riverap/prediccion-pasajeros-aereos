import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CARGAR MODELO Y SCALER
# ==========================================

modelo = joblib.load("modelo_gbr.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# TÍTULO
# ==========================================

st.title("Predicción de Pasajeros Aéreos")

st.write(
    "Aplicación para estimar la cantidad de pasajeros en rutas aéreas internacionales."
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

# ==========================================
# CIUDAD DESTINO
# ==========================================

ciudad_destino = st.selectbox(
    "Ciudad destino",
    [
        'ACAPULCO',
        'ARUBA',
        'ATLANTA',
        'BALBOA',
        'BOSTON',
        'CANCUN',
        'CHICAGO',
        'CIUDAD DE MEXICO',
        'CURACAO',
        'DALAS',
        'DALLAS',
        'FLORIDA',
        'FORT LAUDERDALE',
        'GUADALAJARA',
        'HOUSTON',
        'LEON',
        'LOS ANGELES',
        'MERIDA',
        'MEXICO',
        'MIAMI',
        'MONTERREY',
        'NEW YORK',
        'ORLANDO',
        'PANAMA',
        'PHILADELPHIA',
        'PITTSBURGH',
        'PUERTO PLATA',
        'PUNTA CANA',
        'SANTIAGO DE LOS CABALLEROS',
        'SANTO DOMINGO',
        'TAMPA',
        'WASHINGTON'
    ]
)

# ==========================================
# PAÍS DESTINO
# ==========================================

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

# ==========================================
# BOTÓN PREDICCIÓN
# ==========================================

if st.button("Predecir"):

    columnas = [
        'Año',
        'Mes',
        'Número de Vuelos',
        'Horas Bloque',
        'Sillas Ofrecidas',
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

        'Ciudad Destino_ACAPULCO',
        'Ciudad Destino_ARUBA',
        'Ciudad Destino_ATLANTA',
        'Ciudad Destino_BALBOA',
        'Ciudad Destino_BOSTON',
        'Ciudad Destino_CANCUN',
        'Ciudad Destino_CHICAGO',
        'Ciudad Destino_CIUDAD DE MEXICO',
        'Ciudad Destino_CURACAO',
        'Ciudad Destino_DALAS',
        'Ciudad Destino_DALLAS',
        'Ciudad Destino_FLORIDA',
        'Ciudad Destino_FORT LAUDERDALE',
        'Ciudad Destino_GUADALAJARA',
        'Ciudad Destino_HOUSTON',
        'Ciudad Destino_LEON',
        'Ciudad Destino_LOS ANGELES',
        'Ciudad Destino_MERIDA',
        'Ciudad Destino_MEXICO',
        'Ciudad Destino_MIAMI',
        'Ciudad Destino_MONTERREY',
        'Ciudad Destino_NEW YORK',
        'Ciudad Destino_ORLANDO',
        'Ciudad Destino_PANAMA',
        'Ciudad Destino_PHILADELPHIA',
        'Ciudad Destino_PITTSBURGH',
        'Ciudad Destino_PUERTO PLATA',
        'Ciudad Destino_PUNTA CANA',
        'Ciudad Destino_SANTIAGO DE LOS CABALLEROS',
        'Ciudad Destino_SANTO DOMINGO',
        'Ciudad Destino_TAMPA',
        'Ciudad Destino_WASHINGTON',

        'Pais Destino_ANTILLAS HOLANDESAS',
        'Pais Destino_ESTADOS UNIDOS',
        'Pais Destino_MEXICO',
        'Pais Destino_PANAMA',
        'Pais Destino_REPUBLICA DOMINICANA'
    ]

    # ==========================================
    # CREAR DATAFRAME
    # ==========================================

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
    # ESCALAR
    # ==========================================

    datos_scaled = scaler.transform(datos)

    # ==========================================
    # PREDICCIÓN
    # ==========================================

    pred_log = modelo.predict(datos_scaled)[0]

    # ==========================================
    # VOLVER ESCALA ORIGINAL
    # ==========================================

    pred_real = np.expm1(pred_log)

    # ==========================================
    # MOSTRAR RESULTADO
    # ==========================================

    st.success(
        f"Cantidad estimada de pasajeros: {pred_real:,.0f}"
    )

    # ==========================================
    # DEBUG
    # ==========================================

    st.write("Valor predicho en log:", pred_log)
    st.write("Valor real convertido:", pred_real)