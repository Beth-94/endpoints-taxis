# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import numpy as np
import pickle
import streamlit as st
from streamlit.hello.utils import show_code

def tipo_vehiculo(mes, dia_inicio, hora_inicio, hora_fin, distancia_viaje, ubicacion_inicio, ubicacion_fin, pax, modelo_entrenado):

    try:
        #creamos un array de numpy con las caracteristicas
        datos = np.array([[mes, dia_inicio,hora_inicio,hora_fin,distancia_viaje,ubicacion_inicio,ubicacion_fin,pax]])
        
        # realizamos la predicion usando el modelo
        prediccion = modelo_entrenado.predict(datos)

        #obtenemos el tiempo estimado de llegada
        #tiempo_estimado= obtener_tiempo_estimado(ubicacion_inicio,ubicacion_fin)
        #decodificamos la etiqueta predicha 
        tipovehiculo = cod_num.inverse_transform(prediccion)
    
        return tipovehiculo
    except Exception as e:
        st.error(f'Ocurrio un error al realizar la prediccion: {str(e)}')
        return None

def resultado (tipo_vehiculo_recomendado):
    st.success(f'Tipo de vehículo recomendado: {tipo_vehiculo_recomendado[0]}')
    

# cargamos el modelo entrenado desde el archivo
try:
    with open('modelo_entrenado.pkl', 'rb') as f:
        modelo_entrenado = pickle.load(f)
        #definimos la codificacion de las etiquetas de clase
        cod_num = modelo_entrenado
except Exception as e:
    st.error(f'Ocurrio un error al cargar el modelo: {str(e)}')
    modelo_entrenado = None

def main():
    st.set_page_config(page_title="Prediccion tipo de vehiculo", page_icon="📹")
    st.markdown("# Predicción tipo de vehiculo")
    st.sidebar.header("Predicción tipo de vechiculo")
    st.write(
        """Esta app predice el tipo de vehículo recomendado según las 
        caracteristicas dadas"""
    )

    # creamos lista de opciones para las ubicaciones de inicio y fin
    # Crear una lista de opciones del 1 al 256
    #opciones_ubicaciones = [str(i) for i in range(1, 257)]
    #opc_pax = [str(i) for i in range(1,8)]

    #anio = st.number_input('Año:', min_value=2000, max_value=2100)
    mes = st.number_input('Mes:', min_value=1, max_value=12)
    dia_inicio = st.number_input('Día de inicio:', min_value=1, max_value=31)
    hora_inicio = st.number_input('Hora de inicio:', min_value=0, max_value=23)
    hora_fin = st.number_input('Hora de fin:', min_value=0, max_value=23)
    distancia_viaje = st.text_input('Distancia del viaje (en millas): ')
    ubicacion_inicio =st.number_input('Ubicación de inicio (1 a 265):', min_value=1, max_value= 265)
    ubicacion_fin = st.number_input('Ubicación de fin (1 a 265):', min_value= 1, max_value= 265)
    pax = st.number_input('Número de pasajeros(1 a 7):', min_value=1)

    if st.button('Predecir'):
        try:

            # Realizar la predicción
            tipo_vehiculo_recomendado = tipo_vehiculo(mes, dia_inicio, hora_inicio, hora_fin, distancia_viaje, ubicacion_inicio, ubicacion_fin, pax, cod_num)
            resultado(tipo_vehiculo_recomendado)
            
        except Exception as e:
            st.error(f'Ocurrio un error en los datos de entrada: {str(e)}')
    show_code(tipo_vehiculo)

if __name__=='__main__':
    main()