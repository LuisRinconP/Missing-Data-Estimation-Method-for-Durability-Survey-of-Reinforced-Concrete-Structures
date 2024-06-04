# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:40:57 2023
@author: Luis F. Rincon
    Paper autors: Luis F. Rincon, Bassel Habeeb, Elsa Eustaquio, Ameur Hamami, 
                José Campos e Matos, Yina M. Moscoso, Emilio Bastidas-Arteaga
                
    University of Minho, ISISE, ARISE, Department of Civil Engineering, Guimarães, Portugal
    Laboratory of Engineering Sciences for the Environment (LaSIE - UMR CNRS 7356), La Rochelle University, La Rochelle, France
    Laboratório Nacional de Engenharia Civil, Lisboa, Portugal


TODO JUNTO: PruebaCorre_Temp_Resist.py y Corre_temp_temp_air_llenadoDatos.py(Que tiene mal el nombre)
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
archivo1 = basepath + "/Input_ConcreteResistivityFill.xlsx"
data2 = pd.read_excel(archivo1, sheet_name = "Para_Python")


"""
# Sensor analysis
# Please change the name of the variable Sensores and Temperaturas for the name of the sensor to analyze
# Here are the name of the sensors:
# Sensores = ['MS20R-R15', 'MS20R-R30', 'MN5R-R15', 'MN5R-R30', 'MN20R-R15', 'MN20R-R30', 'JN5R-R15', 'MS5R-R15']
# Temperaturas = ['MS20R-T', 'MS20R-T', 'MN5R-T', 'MN5R-T', 'MN20R-T', 'MN20R-T', 'JN5R-T', 'MS5R-T']
"""
Sensores = ['MN20R-R15']
Temperaturas = ['MN20R-T']

"""
# Subgroups define in the GroupPatternIdentification.py code
# If there is a change in the subgroups obtain from the code GroupPatternIdentification.py
     please change it below
"""    
data = {
    'MS20R-R15': [5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'MS20R-R30': [5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'MN5R-R15': [1776.0, 2299.0, 5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'MN5R-R30': [3311.0, 5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'MN20R-R15': [365.0, 1296.0, 2658.0, 3424.0, 4368.0, 5252.0, np.nan, np.nan, np.nan, np.nan],
    'MN20R-R30': [634.0, 2138.0, 4146.0, 5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'JN5R-R15': [5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'MS5R-R15': [2780.0, 5252.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
}

"""
############## Results
# The results will be exported to a file called 'Output_ConcreteResistivityFill.xslx' 
    in the folder of the python file
"""


# Crear el DataFrame a partir del diccionario
Subgrupos = pd.DataFrame(data)

# Crea una copia del DataFrame original y elimina las columnas "Fecha" y "Fecha.1" de la copia
data_sensors = data2.copy()
data_sensors = data_sensors.drop(["Fecha", "Fecha.1"], axis=1)

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


# Iterar sobre las listas de sensores y temperaturas
for ii in range(len(Sensores)):
    columna1 = Sensores[ii]
    columna2 = Temperaturas[ii]
    columna3 = 'Dias'
    columna4 = 'Air Temp'
    
    # Aquí puedes realizar las operaciones o tareas necesarias utilizando columna1 y columna2
    # Por ejemplo, imprimir los valores en cada iteración
    print(f"Iteración {ii + 1}: columna1 = {columna1}, columna2 = {columna2}")
    
    # Obtener los valores únicos de la columna1 en el DataFrame Subgrupos
    resultados = Subgrupos[columna1].unique().tolist()
    
    ###### Seaparacion de datos del sensor analizado
    if math.isnan(resultados[-1]):
        correct_nan=1
    else:
        correct_nan=0
    
    ##### Calculo de tamalo maximo
    # Inicializar diferencias con el primer valor de resultados
    maximo = float(resultados[0])
    
    # Calcular las diferencias entre elementos consecutivos
    for i in range(len(resultados) - correct_nan - 1):
        diferencia=resultados[i+1]-resultados[i]
        if diferencia > maximo:
            maximo=diferencia
    
    data = [[0 for _ in range(len(resultados) - correct_nan)] for _ in range(int(maximo))]
    
    subgrupo_data = pd.DataFrame()
    subgrupo_dataT = pd.DataFrame()
    subgrupo_data_AirT = pd.DataFrame()
    subgrupo_data_Dias = pd.DataFrame()
    
    for i in range(len(resultados) - correct_nan):
        if i == 0:
            inf = 0
            sup = resultados[i] - 1
        elif i == len(resultados) - 1 - correct_nan:
            inf = resultados[i - 1]
            sup = 5252
        else:
            inf = resultados[i - 1]
            sup = resultados[i] - 1
        
        inf = int(inf)
        sup = int(sup)
        
        subgrupoT = data_sensors[columna2][inf:sup + 1]
        subgrupoT.reset_index(drop=True, inplace=True)
        subgrupo_dataT=pd.concat([subgrupo_dataT, subgrupoT], axis=1)
        
        subgrupo = data_sensors[columna1][inf:sup + 1]
        subgrupo.reset_index(drop=True, inplace=True)
        subgrupo_data=pd.concat([subgrupo_data, subgrupo], axis=1)
        
        subgrupoAirT = data_sensors[columna4][inf:sup + 1]
        subgrupoAirT.reset_index(drop=True, inplace=True)
        subgrupo_data_AirT=pd.concat([subgrupo_data_AirT, subgrupoAirT], axis=1)
        
        subgrupoDias = data_sensors[columna3][inf:sup + 1]
        subgrupoDias.reset_index(drop=True, inplace=True)
        subgrupo_data_Dias=pd.concat([subgrupo_data_Dias, subgrupoDias], axis=1)
        
        # Calcular la correlación entre las columnas 'subgrupo_data' y 'subgrupo_data_AirT'
        correlation = subgrupo.corr(subgrupoAirT)
        print(f"Correlación del grupo {i} del sensor {columna1}: {correlation}")
    
    
        
        
    ###### Calculo de datos faltantes
    
    
    # numero de columna
    num_columnas = len(subgrupo_data.columns)
    
    nombres_columnas = list(range(num_columnas))
    
    # Crear un DataFrame con el mismo tamaño que subgrupo_data
    subgrupo_data_predicted = pd.DataFrame(columns=nombres_columnas, index=subgrupo_data.index)
    
    # Crear un DataFrame vacío para almacenar los tamaños de df_known y df_unknown
    tamanos_dataframe = pd.DataFrame(columns=['size_known', 'size_unknown', 'maximum_gap'])
    
    for i in range(num_columnas):
          # Crear un nuevo DataFrame con las dos columnas deseadas
          # df = pd.DataFrame({'Air Temp': subgrupo_data_AirT.iloc[:, i], 'Dias': subgrupo_data_Dias.iloc[:, i], 'Temp': subgrupo_dataT.iloc[:, i], 'Resist': subgrupo_data.iloc[:, i]})
          df = pd.DataFrame({'Air Temp': subgrupo_data_AirT.iloc[:, i], 'Temp': subgrupo_dataT.iloc[:, i], 'Resist': subgrupo_data.iloc[:, i]})
          df.dropna(subset=['Temp'], inplace=True)
         
          # Separar datos conocidos y desconocidos
          df_known = df.dropna()
          df_unknown = df[df['Resist'].isnull()]
         
          # Obtener los tamaños de df_known y df_unknown en cada iteración
          size_known = len(df_known)
          size_unknown = len(df_unknown)
            
          # Ejemplo de una Serie que contiene los índices donde df['Resist'].isnull() es True
          indices_null = df[df['Resist'].isnull()].index
         
          # Encuentra la longitud máxima de una secuencia continua de índices
          max_length = 0
          current_length = 1 if len(indices_null) > 0 else 0
         
          for j in range(1, len(indices_null)):
              if indices_null[j] == indices_null[j - 1] + 1:
                  current_length += 1
              else:
                  max_length = max(max_length, current_length)
                  current_length = 1
         
          max_length = max(max_length, current_length)
          # print("El tamaño más grande de una secuencia continua de índices es:", max_length)
         
          # Crear un DataFrame con los tamaños actuales
          current_df = pd.DataFrame({'size_known': [size_known], 'size_unknown': [size_unknown], 'maximum_gap': [max_length]})
         
          # Concatenar el DataFrame actual al DataFrame principal
          tamanos_dataframe = pd.concat([tamanos_dataframe, current_df], ignore_index=True)
    
    # print(tamanos_dataframe)
    
    # Graficar 'MS5R-R15' y 'Resist_predicted' con respecto a 'Dias'
    plt.figure(figsize=(10, 6))
    
    Errores = pd.DataFrame(columns=['MSE', 'EAM', 'R²', 'RMSE'])
    
    for i in range(num_columnas):
        # Crear un nuevo DataFrame con las dos columnas deseadas
        # df_X = pd.DataFrame({'Dias': subgrupo_data_Dias.iloc[:, i]})
        df = pd.DataFrame({'Dias': subgrupo_data_Dias.iloc[:, i], 'Air Temp': subgrupo_data_AirT.iloc[:, i], 'Temp': subgrupo_dataT.iloc[:, i], 'Resist': subgrupo_data.iloc[:, i]})
        df.dropna(subset=['Temp'], inplace=True)
        
        # Separar datos conocidos y desconocidos
        df_known = df.dropna()
        df_unknown = df[df['Resist'].isnull()]
        
        
        # Ajustar un modelo GLM
        X = sm.add_constant(df_known[['Air Temp']]) # Agregar una constante y las dos variables predictoras
        y = df_known['Resist']
        
        # Lista de distribuciones y funciones de enlace
        distributions = [sm.families.Gaussian(),  sm.families.Gamma()]
        # distributions = [sm.families.Gamma()] #[sm.families.Gaussian(), sm.families.Poisson(), sm.families.Gamma(), sm.families.Tweedie()]# [sm.families.Gaussian(), sm.families.Poisson(), sm.families.Gamma(), sm.families.Tweedie()]
        #links = [sm.families.links.identity, sm.families.links.log, sm.families.links.logit, sm.families.links.probit]
        links = [sm.families.links.identity, sm.families.links.log]
        
        # Crear una nueva columna en el DataFrame para almacenar los valores predichos
        df['sensor_Predicted'] = np.nan
        
        # Inicializar una matriz vacía para almacenar los valores
        num_distributions = len(distributions)  # Suponiendo que distributions es una lista de distribuciones
        num_metrics = 6  # Número de métricas a almacenar
        
        # Crear una matriz para almacenar los resultados
        result_matrix = np.full((num_distributions, num_metrics), np.nan)
        
        # for i, distribution in enumerate(distributions):
            # print(i)
            # model = sm.GLM(y, X, family=distribution)
            # results = model.fit()
            # # Imprimir información sobre la distribución y función de enlace utilizadas
            # distribution_name = distribution.__class__.__name__
            # # print(f"Combination: Distribution={distribution_name}")
            
            # # Deviance
            # deviance_i=results.deviance
            
            # # Likelihood
            # log_likelihood = results.llf
            
            # # Calcular la deviance nula (null deviance)
            # null_deviance = results.null_deviance
            
            # # Calcular la deviance residual (residual deviance)
            # residual_deviance = results.deviance
            
            # # Calcular el R-cuadrado de McFadden
            # pseudo_r2_mcfadden = 1 - (residual_deviance / null_deviance)
            
            # # Obtén el valor del AIC
            # aic = results.aic
            
            # # Obtén el valor del BIC
            # bic = results.bic
            
            # print(i)
            # # Almacenar los valores en la matriz
            # result_matrix[i] = [deviance_i, log_likelihood, residual_deviance, pseudo_r2_mcfadden, aic, bic]
    
        
        ############ AQUI HAY QUE ESCOGER LA FAMILIA
        # Ajustar el modelo GLM con la familia Gaussiana y el enlace identidad
        model = sm.GLM(y, X, family=sm.families.Gaussian())
        results = model.fit()        
        
        df_temp_with_constant = sm.add_constant(df_unknown[['Air Temp']])
        
        # Predecir valores de Variable_A para los datos desconocidos
        predicted_values = results.predict(df_temp_with_constant)
        
        # Llenar los valores faltantes en la nueva columna
        df.loc[df['Resist'].isnull(), 'sensor_Predicted'] = predicted_values.values.copy()
        
        # Agregar la columna 'sensor_Predicted' del DataFrame 'df' a 'subgrupo_data_predicted' en la posición 'i'
        subgrupo_data_predicted[i] = df['sensor_Predicted']
        
        
        # Supongamos que tienes df_known con una columna 'Dias' que inicia en 1 y df_sensor con tus datos
        fecha_inicial = pd.to_datetime('2006-07-04')
        
        # Crear un nuevo DataFrame solo con las fechas
        df_fechas_known = pd.DataFrame({'Fechas': fecha_inicial + pd.to_timedelta(df_known['Dias'] - 1, unit='D')})
        
        # Crear un nuevo DataFrame solo con las fechas
        df_fechas_unknown = pd.DataFrame({'Fechas': fecha_inicial + pd.to_timedelta(df_unknown['Dias'] - 1, unit='D')})
        
        
        # Gráfico de puntos con marcadores 'o' (círculos) de color azul ('b')  y tamaño de marcador reducido
        plt.scatter(df_fechas_known['Fechas'], df_known['Resist'], label='Resistivity', marker='o', color='b', s=5)
    
        # Gráfico de 'Resist_predicted'
        plt.scatter(df_fechas_unknown['Fechas'], predicted_values, label='Resist_predicted', s=5)
        
        plt.xlabel('Date')
        plt.ylabel('Concrete Resistivity (Ohm)')
        plt.title(f"Gráfico de {columna1}")
        plt.grid(True)
        
        
        
        # Obtener los residuos
        residuals = results.resid_response
        
        # Calcular el Error Cuadrático Medio (ECM)
        mse = np.mean(residuals ** 2)
        # print("Error Cuadrático Medio (ECM):", mse)
        
        # Calcular el Error Absoluto Medio (EAM)
        mae = np.mean(np.abs(residuals))
        # print("Error Absoluto Medio (EAM):", mae)
        
        # Obtener los valores predichos
        predicted_values2 = results.predict()
        
        # Valores reales
        true_values = df_known['Resist']  # Asumiendo que 'Resist' es tu variable dependiente en el DataFrame 'df'
        
        # Calcular los residuos
        residuals = true_values - predicted_values2
        
        # Calcular SST
        SST = np.sum((true_values - np.mean(true_values)) ** 2)
        
        # Calcular SSR
        SSR = np.sum(residuals ** 2)
        
        # Calcular R²
        r_squared = 1 - (SSR / SST)
        # print("R-cuadrado (R²):", r_squared)
        
        # Calcular el error relativo
        error_absoluto = np.mean(np.abs(true_values - predicted_values2))
        error_relativo = error_absoluto / true_values
        erm = np.mean(error_relativo)
        # Imprimir el Error Relativo Medio (ERM)
        # print("Error Relativo Medio (ERM):", erm)
        
        # Calcular el Error Cuadrático Medio (ECM)
        mse = np.mean(residuals ** 2)
        
        # Calcular el RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
    
    
        # Crear un DataFrame con los tamaños actuales
        current_df = pd.DataFrame({'MSE': [mse], 'EAM': [mae], 'R²': [r_squared], 'RMSE': [rmse]})
         
        # Concatenar el DataFrame actual al DataFrame principal
        Errores = pd.concat([Errores, current_df], ignore_index=True)
        

    
    # print(Errores)
    
    # Obtener el DataFrame correspondiente al sensor actual
    sensor_df = dataframes_dict[columna1]
        
    # Concatenar el DataFrame de resultados al DataFrame correspondiente al sensor actual
    dataframes_dict[columna1] = pd.concat([tamanos_dataframe, Errores], axis=1)
    
    result = pd.concat([tamanos_dataframe, Errores], axis=1)
    
    # Mostrar el resultado
    print(Errores)
    print(result)
    
    plt.show()


    ######### Juntar los resultados en un solo vector
    
    infe = 0
    
    Resist_predicted = pd.DataFrame({columna1: [np.nan] * 5252})
    Resist_predicted[columna1].fillna(0, inplace=True)
    # = pd.DataFrame({'columna1': [np.nan] * 5251})
    # = pd.DataFrame({'columna1': [np.nan] * 5251})
    
    for i in range(len(resultados) - correct_nan):
        if i == 0:
            infe = 0
            sup = resultados[i] - 1
        else:
            infe = resultados[i - 1]
            sup = resultados[i] - 1
        
        infe = int(infe)
        sup = int(sup)
        
        
        # Obtener el número de columnas en el DataFrame
        n_columnas = len(subgrupo_data.columns)
        
        # Cambiar los nombres de las columnas a números de 0 a n_columnas-1
        subgrupo_data.columns = range(n_columnas)
        
        columna_por_copiar = subgrupo_data_predicted.iloc[:int(sup-infe+1), i].copy()   
        # print(len(columna_por_copiar))
        
        Resist_predicted.loc[infe:infe+len(columna_por_copiar) - 1, columna1] = columna_por_copiar.values
    



n_rows, n_cols = subgrupo_data.shape
Results_subgroups = pd.DataFrame(np.nan, index=range(n_rows), columns=range(n_cols*3))

for i in range(n_cols):
    Results_subgroups.iloc[:, i*3] = subgrupo_data_Dias.iloc[:, i]
    Results_subgroups.iloc[:, i*3+1] = subgrupo_data.iloc[:, i]
    Results_subgroups.iloc[:, i*3+2] = subgrupo_data_predicted.iloc[:, i]
    Results_subgroups.rename(columns={i*3:'Days'}, inplace=True)
    Results_subgroups.rename(columns={i*3+1:'Measured'}, inplace=True)
    Results_subgroups.rename(columns={i*3+2:'Filled'}, inplace=True)
    
    
Results_subgroups.to_excel('Output_ConcreteResistivityFill.xlsx', sheet_name='Subgroups')


