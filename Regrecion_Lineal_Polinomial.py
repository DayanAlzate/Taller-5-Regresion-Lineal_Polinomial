#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:35:38 2022
@author: dayan alzate hernandez
Correo Institucional: dayan.alzate@upb.edu.co
Id:000502226
"""

#from sklearn import linear_model 
#from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score

## convertimos el archivo csv en un data frame 
dato= pd.read_csv('AirQualityUCI.csv',sep=';')
## selecionamos los datos X y Y para relalizar nuestra regresion 
#  obtenemos el 20% de los datos para hacer las pruebas 
x= dato["NO2(GT)"].head(200)
y= dato["NOx(GT)"].head(200)
    
print ("Regresion Lineal ")
# Graficamos x con respecto a Y 
def grafico1_RLineal():
    plt.scatter(x,y)
    plt.xlabel("NO2(GT)")
    plt.ylabel("NOx(GT)")
    plt.title("NO2 vs NOx")

grafico1_RLineal()

# calculamos nuestro valor de R 
def R():
    slope,intercept,r,p,std_err = stats.linregress(x,y)
    return slope,intercept,r,p,std_err

# Calculamos los valores de la regersion lineal para poder graficar 
def regresion(x):
    slope,intercept,r,p,std_err=R()
    return slope * x + intercept

#graficamos la linea de la regresion 
def grafico2_RLineal():
    reg_model = list(map(regresion,x))
    plt.plot(x,reg_model)
    plt.show()
    
grafico2_RLineal()
slope,intercept,r,p,std_err=R()


print("Valor de R: "+str(r))
print ("-------------------------------------------------------------\n")

print ("\n Regresion Polinomial ")


def Regresion_Polinomial():
    poli_model = np.poly1d(np.polyfit(x,y,6)) 
    poli_line = np.linspace(2,200,100)
    
    plt.scatter(x,y)
    plt.plot(poli_line,poli_model(poli_line))
    plt.show()
    print(r2_score(y,poli_model(x)))
    
Regresion_Polinomial()








    
    
    

                                                                                                                                                