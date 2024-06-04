# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:40:57 2023
@author: Luis F. Rincon
    Paper autors: Luis F. Rincon, Bassel Habeeb, Elsa Eustaquio, Ameur Hamami, 
                José Campos e Matos, Yina M. Moscoso, Emilio Bastidas-Arteaga
                
    University of Minho, ISISE, ARISE, Department of Civil Engineering, Guimarães, Portugal
    Laboratory of Engineering Sciences for the Environment (LaSIE - UMR CNRS 7356), La Rochelle University, La Rochelle, France
    Laboratório Nacional de Engenharia Civil, Lisboa, Portugal

"""
# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
from matplotlib.animation import FuncAnimation
import math
import random

"""
############## Data read
# Please change the location of the data file in the variable basepath
    Remember to use / instead of \ in the direction of the file
""" 

basepath = "D:/OneDrive Uminho/OneDrive - Universidade do Minho/0. Doctorado U.Mihno/11. Congresos_Articulos/7. Articulo sensores/Structural Health Monitoring - An International journal/GitHub"
archivo1 = basepath + "/Input_GroupPatternIdentification.xlsx"
data = pd.read_excel(archivo1, sheet_name = "Para_Python")

data_sensors = data.copy()
data_sensors = data_sensors.drop(["Fecha", "Fecha.1"], axis=1)

"""
# Sensor analysis
# Please change the name of the variable Sensores and Temperaturas for the name of the sensor to analyze
# Here are the name of the sensors:
# Sensores = ['MS20R-R15', 'MS20R-R30', 'MN5R-R15', 'MN5R-R30', 'MN20R-R15', 'MN20R-R30', 'JN5R-R15', 'MS5R-R15']
# Temperaturas = ['MS20R-T', 'MS20R-T', 'MN5R-T', 'MN5R-T', 'MN20R-T', 'MN20R-T', 'JN5R-T', 'MS5R-T']
"""
Sensores = ['MN20R-R30']
Temperaturas = ['MN20R-T']

"""
############## Results
# The results will be exported to a file called 'Output_GroupPatternIdentification.xslx' 
    in the folder of the python file
    The column i on the results represent the line where a new subgroup start
"""




# Creando un diccionario para almacenar los DataFrames por cada sensor
dataframes_dict = {}

# Creación de los DataFrames para cada sensor con las columnas especificadas
for sensor in Sensores:
    data = {
        'ECM': [],
        'EAM': [],
        'R²': [],
        'ERM': []
    }
    # Se crea el DataFrame para el sensor actual con datos vacíos
    df = pd.DataFrame(data)
    
    # Se agrega el DataFrame al diccionario usando el nombre del sensor como clave
    dataframes_dict[sensor] = df


# Crear el DataFrame Subgrupos con las columnas de Sensores y llenarlo con NaN
num_filas = 10
Subgrupos = pd.DataFrame({sensor: [np.nan] * num_filas for sensor in Sensores})

# Iterar sobre las listas de sensores y temperaturas
for ii in range(len(Sensores)):
    columna1 = Sensores[ii]
    columna2 = Temperaturas[ii]
    columna3 = 'Dias'
    columna4 = 'Air Temp'
    
    print(columna1)
    
    nuevo_inicio = 0
    fila_huecos = 0
    
    resultados = pd.DataFrame(columns=['i','diferencia1','diferencia2','correlacion1','correlacion2'])
    
    df_resultados = pd.DataFrame(columns=['i', 'ultimo_valor_conocido', 'ultimo_valor_predicho'])
    df_resultados2 = pd.DataFrame(columns=['i', 'ultimo_valor_conocido2', 'ultimo_valor_predicho2'])
    
    for i in range(2, len(data_sensors[columna1]), 1):
        if math.isnan(data_sensors[columna1][i]):
            continue  # Esta línea debe estar indentada correctamente
    
        valor1 = data_sensors[columna1][nuevo_inicio:i]
        valor2 = data_sensors[columna2][nuevo_inicio:i]
        valor3 = data_sensors[columna3][nuevo_inicio:i]
        valor3 = valor3.astype(float)
        
        # Eliminar NaN de valor1 y valor2 al mismo tiempo
        valores_no_nulos = valor1.notnull() & valor2.notnull() & valor3.notnull()
        valor1 = valor1[valores_no_nulos]
        valor2 = valor2[valores_no_nulos]
        valor3 = valor3[valores_no_nulos]
        
        if len(valor1) >= 365:
            # ENTRE RESISTIVIDAD Y TEMPERATURA
            # Sensor específico que deseas analizar
            sensor = 'sensor'
            
            # Crear un nuevo DataFrame con las dos columnas deseadas
            df = pd.DataFrame({'Temp': valor2, 'sensor': valor1})
            
            # Obtener el índice del último elemento en la columna 'sensor'
            ultimo_indice = df.index[-2:]
            # Cambiar el último valor de la columna 'sensor' por NaN
            df.loc[ultimo_indice, 'sensor'] = np.nan
            
            # Separar datos conocidos y desconocidos
            df_known = df.dropna()
            df_unknown = df[df[sensor].isnull()]
            
            # Verificar si los últimos dos valores de la columna 'Temp' son iguales en df_unknown
            if df_unknown['Temp'].iloc[-1] == df_unknown['Temp'].iloc[-2]:
                # Agregar 0.001 al penúltimo valor
                df_unknown.at[df_unknown.index[-2], 'Temp'] += 0.001
    
            # Ajustar un modelo GLM
            X = sm.add_constant(df_known['Temp'])  # Agregar una constante
            y = df_known[sensor]
    
            # Crear una nueva columna en el DataFrame para almacenar los valores predichos
            df['sensor_Predicted'] = np.nan
    
            # Crear una instancia del enlace identidad
            identity_link = sm.families.links.identity()
            
            # Crear una instancia de la familia Gaussiana con el enlace identidad
            gaussian_family = sm.families.Gaussian(link=identity_link)
            
            # Ajustar el modelo GLM con la familia Gaussiana y el enlace identidad
            model = sm.GLM(y, X, family=gaussian_family)
            results = model.fit()
            
            df_temp_with_constant = sm.add_constant(df_unknown['Temp'])
            
            # Predecir valores de Variable_A para los datos desconocidos
            predicted_values = results.predict(df_temp_with_constant)
    
            # Llenar los valores faltantes en la nueva columna
            df.loc[df[sensor].isnull(), 'sensor_Predicted'] = predicted_values.values.copy()
    
            # Obtener el último valor conocido de valor1
            ultimo_valor_conocido = valor1.iloc[-1]
            
            # Obtener el último valor predicho
            ultimo_valor_predicho = predicted_values.values[-1]
            
            # Crear un DataFrame temporal con estos valores
            nuevo_registro = pd.DataFrame({'i': [i],
                                            'ultimo_valor_conocido': [ultimo_valor_conocido],
                                            'ultimo_valor_predicho': [ultimo_valor_predicho]})
        
            # Concatenar este registro al DataFrame de resultados
            df_resultados = pd.concat([df_resultados, nuevo_registro], ignore_index=True)
            
            # Diferencia
            diferencia = abs(ultimo_valor_conocido - ultimo_valor_predicho) / abs(ultimo_valor_conocido)
            correlacion1 = valor1.corr(valor2)
            
            
            
            
            # ENTRE RESISTIVIDAD Y FECHA ############################################
           
            # Crear un nuevo DataFrame con las dos columnas deseadas
            df2 = pd.DataFrame({'Dias': valor3, 'sensor': valor1})
            
            # Obtener el índice del último elemento en la columna 'sensor'
            ultimo_indice = df2.index[-2:]
            # Cambiar el último valor de la columna 'sensor' por NaN
            df2.loc[ultimo_indice, 'sensor'] = np.nan
            
            # Separar datos conocidos y desconocidos
            df_known2 = df2.dropna()
            df_unknown2 = df2[df2[sensor].isnull()]
    
            # Ajustar un modelo GLM
            X2 = sm.add_constant(df_known2['Dias'])  # Agregar una constante
            y2 = df_known2[sensor]
    
            # Crear una nueva columna en el DataFrame para almacenar los valores predichos
            df2['sensor_Predicted'] = np.nan
    
            # Crear una instancia del enlace identidad
            identity_link = sm.families.links.identity()
            
            # Crear una instancia de la familia Gaussiana con el enlace identidad
            gaussian_family = sm.families.Gaussian(link=identity_link)
            
            # Ajustar el modelo GLM con la familia Gaussiana y el enlace identidad
            model2 = sm.GLM(y2, X2, family=gaussian_family)
            results2 = model2.fit()        
            
            df_temp_with_constant2 = sm.add_constant(df_unknown2['Dias'])
            
            # Predecir valores de Variable_A para los datos desconocidos
            predicted_values2 = results2.predict(df_temp_with_constant2)
    
            # Llenar los valores faltantes en la nueva columna
            df2.loc[df2[sensor].isnull(), 'sensor_Predicted'] = predicted_values2.values.copy()
    
            # Obtener el último valor conocido de valor1
            ultimo_valor_conocido2 = valor1.iloc[-1]
            
            # Obtener el último valor predicho
            ultimo_valor_predicho2 = predicted_values2.values[-1]
            
            
            # Crear un DataFrame temporal con estos valores
            nuevo_registro = pd.DataFrame({'i': [i],
                                            'ultimo_valor_conocido2': [ultimo_valor_conocido2],
                                            'ultimo_valor_predicho2': [ultimo_valor_predicho2]})
            
            # Concatenar este registro al DataFrame de resultados
            df_resultados2 = pd.concat([df_resultados2, nuevo_registro], ignore_index=True)
            
            # Diferencia
            diferencia2 = abs(ultimo_valor_conocido2 - ultimo_valor_predicho2) / abs(ultimo_valor_conocido2)
            correlacion2 = valor1.corr(valor3)

            if diferencia2 < 1.56 and diferencia < 0.8:
                pass # La correlacion es adecuada
            else:
                # print(i-1)
                Subgrupos.loc[fila_huecos, columna1] = i-1
                fila_huecos = fila_huecos + 1
                # print(diferencia)
                # print(diferencia2)
                nuevo_inicio = i
                nueva_fila = pd.DataFrame({'i': [i-1], 'diferencia1': [diferencia], 'diferencia2': [diferencia2], 'correlacion1': [correlacion1], 'correlacion2': [correlacion2]})
                resultados = pd.concat([resultados, nueva_fila], ignore_index=True)

        else:
            pass # Pasa al siguiente ciclo por tener pocos datos
        
        if i+1==len(data_sensors[columna1]):
            nueva_fila = pd.DataFrame({'i': [i+1], 'diferencia1': [diferencia], 'diferencia2': [diferencia2], 'correlacion1': [correlacion1], 'correlacion2': [correlacion2]})
            resultados = pd.concat([resultados, nueva_fila], ignore_index=True)        
    
    # print(i+1)
    # Llenar el valor 1 en la fila fila_huecos y la columna correspondiente a columna1
    Subgrupos.loc[fila_huecos, columna1] = i+1
    print(resultados)
    # print(diferencia2)
    
df_columna_i = resultados[['i']]
df_columna_i.to_excel('Output_GroupPatternIdentification.xlsx')