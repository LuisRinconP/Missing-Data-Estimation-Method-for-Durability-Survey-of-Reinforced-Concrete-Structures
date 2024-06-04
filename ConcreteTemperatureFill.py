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
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
############## Data read
# Please change the location of the data file in the variable basepath
    Remember to use / instead of \ in the direction of the file
""" 
basepath = "D:/OneDrive Uminho/OneDrive - Universidade do Minho/0. Doctorado U.Mihno/11. Congresos_Articulos/7. Articulo sensores/Structural Health Monitoring - An International journal/GitHub"
archivo1 = basepath + "/Input_ConcreteTemperatureFill.xlsx"
dataTemp = pd.read_excel(archivo1, sheet_name = "Para_Python")

"""
############## Sensor to analyze
# Please change the name of the variable sensor for the name of the sensor to analyze
# Here are the name of the sensors:
# Sensor = 'JN5R-T', 'MS5R-T', 'MS20R-T', 'MN5R-T', 'MN20R-T'
"""
sensor = 'MS5R-T' 

"""
############## Results
# The results will be exported to a file called 'Output_ConcreteTemperatureFill.xslx' 
    in the folder of the python file
"""



# Correlacion
corrmat = dataTemp.corr()
# Configura el tamaño de la figura
plt.figure(figsize=(10, 6))
# Crea la gráfica de matriz de correlación
sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt=".2f", square=True)
# Personaliza la gráfica
plt.title("Matriz de Correlación")
plt.show()

# Crear un nuevo DataFrame con las dos columnas deseadas
df = pd.DataFrame({'Air Temp': dataTemp['Air Temp'], sensor: dataTemp[sensor]})

# Separar datos conocidos y desconocidos
df_known = df.dropna()
df_unknown = df[df[sensor].isnull()]

# Ajustar un modelo GLM
X = sm.add_constant(df_known['Air Temp'])
y = df_known[sensor]

# Lista de distribuciones y funciones de enlace
distributions = [sm.families.Gaussian()] 
links = [sm.families.links.identity]

# Crear una nueva columna en el DataFrame para almacenar los valores predichos
df['sensor_Predicted'] = np.nan

for distribution in distributions:
    for link in links:
        model = sm.GLM(y, X, family=distribution, link=link)
        results = model.fit()
        # Imprimir información sobre la distribución y función de enlace utilizadas
        distribution_name = distribution.__class__.__name__
        link_name = link.__name__
        print(f"Combination: Distribution={distribution_name}, Link={link_name}")
        
        print("Deviance:", results.deviance)
        # Likelihood
        log_likelihood = results.llf
        print("Log-likelihood:", log_likelihood)
        # print(results.summary())
        
        # Calcular la deviance nula (null deviance)
        null_deviance = results.null_deviance
        
        # Calcular la deviance residual (residual deviance)
        residual_deviance = results.deviance
        
        # Calcular el R-cuadrado de McFadden
        pseudo_r2_mcfadden = 1 - (residual_deviance / null_deviance)
        
        print("Pseudo R-cuadrado (McFadden):", pseudo_r2_mcfadden)
        
        # Obtén el valor del AIC
        aic = results.aic
        
        # Obtén el valor del BIC
        bic = results.bic
        
        # Imprime los valores del AIC y el BIC
        print("AIC:", aic)
        print("BIC:", bic)
      
        # Predecir valores de Variable_A para los datos desconocidos
        predicted_values = results.predict(sm.add_constant(df_unknown['Air Temp']))
        
        # Llenar los valores faltantes en la nueva columna
        df.loc[df[sensor].isnull(), 'sensor_Predicted'] = predicted_values.values.copy()

        # Crear un gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(df_known[sensor], df_known['Air Temp'], label='Measured data', color='blue', alpha=0.7)
        plt.scatter(predicted_values, df_unknown['Air Temp'], label='Data filling', color='red', marker='x', s=50)
        plt.xlabel(sensor)
        plt.ylabel('Air Temp')
        plt.legend()
        plt.title(f'Measured data vs. Data filling - {sensor} ({distribution_name}, {link_name})')
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()

         # Calcular los residuos
        residuals = results.resid_response

        # Gráfico de residuos vs. valores ajustados (homocedasticidad)
        plt.figure(figsize=(12, 6))
        plt.scatter(results.fittedvalues, residuals)
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Residuos')
        plt.title('Gráfico de Residuos vs. Valores Ajustados')
        plt.axhline(0, color='red', linestyle='dashed')
        plt.grid(True)
        plt.show()

        # Gráfico de cuantiles-cuantiles (QQ) (normalidad de los residuos)
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'Gráfico Cuantiles-Cuantiles (QQ) de Residuos ({distribution_name}, {link_name})')
        plt.grid(True)
        plt.show()
        
        # Huecos artificiales
        zona_conocida_1 = df_known.iloc[148:278]  # Primera zona conocida
        zona_conocida_2 = df_known.iloc[1438:1568]  # Segunda zona conocida
        zona_conocida_3 = df_known.iloc[2261:2391]  # Tercera zona conocida
        # Predecir los valores del sensor en la zona conocida
        # Predecir los valores del sensor en la primera zona conocida
        valores_predichos_zona_1 = results.predict(sm.add_constant(zona_conocida_1['Air Temp']))
        # Predecir los valores del sensor en la segunda zona conocida
        valores_predichos_zona_2 = results.predict(sm.add_constant(zona_conocida_2['Air Temp']))
        # Predecir los valores del sensor en la tercera zona conocida
        valores_predichos_zona_3 = results.predict(sm.add_constant(zona_conocida_3['Air Temp']))        
        # Grafica zona 1
        plt.figure(figsize=(10, 6))
        plt.plot(zona_conocida_3.index, zona_conocida_3[sensor], label='Valores Originales (Zona 1)', marker='o', color='blue')
        plt.plot(zona_conocida_3.index, valores_predichos_zona_3, label='Valores Predichos (Zona 1)', marker='x', color='red')
        plt.xlabel('Índice de la Zona 1')
        plt.ylabel('Valor del Sensor')
        plt.legend()
        plt.title('Comparación de Valores Originales y Predichos en la Zona 1')
        plt.grid(True)
        
        # Mostrar el gráfico
        plt.show()
df.to_excel('Output_ConcreteTemperatureFill.xlsx')