# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:17:49 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Documentación y ayuda contextual
st.sidebar.markdown("### Ayuda y Documentación")
st.sidebar.markdown("""
    Esta aplicación de monitoreo de compresores te permite visualizar y analizar datos de presión, temperatura y vibración.
    
    **Instrucciones:**
    
    1. Utiliza los filtros de fecha en el panel lateral izquierdo para seleccionar el rango de fechas que deseas analizar.
    2. Ajusta los umbrales de presión, temperatura y vibración para detectar anomalías.
    3. Observa las estadísticas descriptivas para comprender mejor los datos.
    4. Exporta los datos filtrados a un archivo CSV si es necesario.
    5. ¡Explora y disfruta del dashboard!
""")




# Título del Dashboard
st.title('Dashboard de Monitoreo de Compresores')

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('datos_compresor.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Definir umbrales de presión, temperatura y vibración
threshold_pressure = st.sidebar.slider('Umbral de Presión', min_value=0, max_value=100, value=60)
threshold_temperature = st.sidebar.slider('Umbral de Temperatura', min_value=0, max_value=150, value=90)
threshold_vibration = st.sidebar.slider('Umbral de Vibración', min_value=0.0, max_value=1.0, value=0.7)

# Alerta de umbrales superados
df = load_data()
if (df['presion'] > threshold_pressure).any() or (df['temperatura'] > threshold_temperature).any() or (df['vibracion'] > threshold_vibration).any():
    st.warning('¡Se han superado los umbrales predefinidos para presión, temperatura o vibración!')

# Visualización de Datos
st.subheader('Datos Recientes')
st.write(df)

# Selección de fecha para el análisis
st.sidebar.subheader('Filtros')
start_date = st.sidebar.date_input('Fecha de inicio', df['timestamp'].min().date())
end_date = st.sidebar.date_input('Fecha de fin', df['timestamp'].max().date())

# Filtrar los datos por la fecha seleccionada
filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Aplicar filtros de umbrales a los datos
anomalies = filtered_df[(filtered_df['presion'] > threshold_pressure) |
                        (filtered_df['temperatura'] > threshold_temperature) |
                        (filtered_df['vibracion'] > threshold_vibration)]

filtered_df = filtered_df[(filtered_df['presion'] <= threshold_pressure) &
                          (filtered_df['temperatura'] <= threshold_temperature) &
                          (filtered_df['vibracion'] <= threshold_vibration)]

# Exportar datos filtrados a CSV
if not filtered_df.empty:
    if st.button('Exportar datos filtrados a CSV'):
        filtered_df.to_csv('datos_filtrados.csv', index=False)
        st.success('¡Los datos filtrados se han exportado exitosamente!')

# Implementación del modelo predictivo
if not filtered_df.empty:
    # Dividir los datos en características (X) y variable objetivo (y)
    X = filtered_df['timestamp'].astype(np.int64).values.reshape(-1, 1)
    y = filtered_df['presion'].values
    
    # Entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    
    # Realizar predicciones
    predictions = model.predict(X)
    
    # Calcular el error cuadrático medio
    mse = mean_squared_error(y, predictions)
    
    # Mostrar el rendimiento del modelo
    st.subheader('Rendimiento del Modelo')
    st.write('Error Cuadrático Medio:', mse)
    
    # Gráfico de predicciones vs observaciones reales
    fig_predictions = px.scatter(x=filtered_df['timestamp'], y=filtered_df['presion'], title='Predicciones vs Observaciones de Presión', trendline="lowess")
    fig_predictions.update_traces(line=dict(color='blue', width=1))
    fig_predictions.update_xaxes(title='Fecha y Hora')
    fig_predictions.update_yaxes(title='Presión')
    st.plotly_chart(fig_predictions)

# Gráfico de presión
st.subheader('Presión del Compresor')
fig_pressure = px.line(filtered_df, x='timestamp', y='presion', title='Presión a lo largo del tiempo')
fig_pressure.update_traces(line=dict(color='blue', width=1))
fig_pressure.update_xaxes(title='Fecha y Hora', rangeslider_visible=True)
fig_pressure.update_yaxes(title='Presión')
st.plotly_chart(fig_pressure)

# Gráfico de temperatura
st.subheader('Temperatura del Compresor')
fig_temperature = px.line(filtered_df, x='timestamp', y='temperatura', title='Temperatura a lo largo del tiempo')
fig_temperature.update_traces(line=dict(color='red', width=1))
fig_temperature.update_xaxes(title='Fecha y Hora', rangeslider_visible=True)
fig_temperature.update_yaxes(title='Temperatura')
st.plotly_chart(fig_temperature)

# Gráfico de vibración
st.subheader('Vibración del Compresor')
fig_vibration = px.line(filtered_df, x='timestamp', y='vibracion', title='Vibración a lo largo del tiempo')
fig_vibration.update_traces(line=dict(color='green', width=1))
fig_vibration.update_xaxes(title='Fecha y Hora', rangeslider_visible=True)
fig_vibration.update_yaxes(title='Vibración')
st.plotly_chart(fig_vibration)

# Estadísticas descriptivas
st.subheader('Estadísticas Descriptivas')
st.write(filtered_df.describe())

# Alerta de anomalías
if not anomalies.empty:
    st.warning('¡Se han detectado anomalías en los datos!')
    
    # Análisis de causa raíz para anomalías detectadas
    st.subheader('Análisis de Causa Raíz para Anomalías Detectadas')
    
    # Identificar las características de las anomalías
    anomaly_features = anomalies[['presion', 'temperatura', 'vibracion']]
    
    # Calcular estadísticas descriptivas de las características de las anomalías
    anomaly_stats = anomaly_features.describe()
    
    st.write('Estadísticas Descriptivas de las Anomalías:')
    st.write(anomaly_stats)
    
    # Posibles causas raíz basadas en las características de las anomalías
    possible_root_causes = []
    
    # Ejemplo de análisis: comparar las anomalías con los umbrales
    for index, row in anomaly_stats.iterrows():
        if row['presion'] > threshold_pressure:
            possible_root_causes.append('Presión alta')
        if row['temperatura'] > threshold_temperature:
            possible_root_causes.append('Temperatura alta')
        if row['vibracion'] > threshold_vibration:
            possible_root_causes.append('Vibración alta')
    
    st.write('Posibles Causas Raíz:')
    st.write(possible_root_causes)

# Mostrar anormalidades detectadas
st.subheader('Alertas de Anormalidades')
st.write(anomalies)

# Créditos del creador
st.sidebar.markdown("---")
st.sidebar.text("Creado por:")
st.sidebar.markdown("<span style='color: orange;'>Javier Horacio Pérez Ricárdez</span>", unsafe_allow_html=True)
