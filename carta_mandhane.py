#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:39:46 2022

@author: fluidos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import requests
from PIL import Image
import io
from sklearn.cluster import mean_shift
import sys 
import os
import random
#sys.path.append(os.path.abspath("/Users/lffp/Documents/posgrado/materia_bifasico/4 Modelado de Transiciones"))
#from Modelado_Transiciones import Parametros as t_Ducklker

#Variables Auxilixares
pi = np.pi
power = np.power
sqrt = np.sqrt
arcos = np.arccos


#Función Principal
def __main__():

    #Parametros del problema Inicial: Tubería vertical
    
    theta = 90.
    d = 2.54
    w_l = 1.
    rho_l = 1000
    mu_l = 1.e-3
    w_g = 0.01135
    rho_g = 1.6
    mu_g = 0.02e-3
    T = 25
    
    Parametros(d, rho_l, w_l, mu_l, rho_g, w_g, mu_g, T, theta=0, g=9.81)
  
def Parametros(d, rho_l, w_l, mu_l, rho_g, w_g, mu_g, T, theta=0, g=9.81):
    """
    Entrada donde se introducen los parámetros del problema e imprime en pantalla los valores
    deseados adimensionales

    Parameters
    ----------
    d : float
        Diametro del tubo en cm.
    rho_l : float
        densidad del liquido en kg/m^3.
    w_l : float
        flujo masico del liquido en kg/s.
    mu_l : float
        viscocidad del liquido en kg/m*s.
    rho_g : float
        densidad del gas en kg/m^3.
    w_g : float
        flujo masico del gas en kg/s.
    mu_g : float
        viscocidad del gas en kg/m*s.
    T : float
        Temperatura del sistema en grados celcius
    theta : float, optional
        Angulo de inclinación de la tuberia en grados. The default is 0.
    g : float, optional
        modulo de la gravedad en m/s^2. The default is 9.81.
        
    Returns
    -------
    None.
    
    """
    
    '''----------------------*
    | PASAR VARIABLES A S.I. |
    *----------------------'''
    
    #Pasar el diametro a m
    d = d/100.
    
    #Pasar el angulo a radiantes
    theta = theta*pi/180.
    
    #Pasar la temperatura a Kelvin
    T = T + 273.15
        
    
    '''----------------*
    | CALCULOS PREVIOS |
    *----------------'''
    
    #Area transversal de la tuberia
    A = 0.25*pi*d*d
    
    #Calidad
    x = w_g/(w_g + w_l)
    
   
    
        
    lim = [1e-2, 1e2, 1e-3, 10]
    #lim1 = [0.01, 100]

    lista_vel, lista_GX, lista_VhX = lista_puntos(rho_l, rho_g, lim)
    

    #Carta velocidad vs velocidad, original
    Mandhane(w_l, w_g, x, A, rho_l, rho_g, lista_vel, lim, True, False)
    
    #Carta G vs X
    velocidad_mezcla = False
    Mandhane(w_l, w_g, x, A, rho_l, rho_g, lista_GX, lim, False, velocidad_mezcla)
    
    #Carta velocidad mezcla vs x
    velocidad_mezcla = True
    Mandhane(w_l, w_g, x, A, rho_l, rho_g, lista_VhX, lim, False, velocidad_mezcla)
 
    
    print(' ')
    print('************* VEL. SUP. GAS vs VEL. SUP. LIQ. *************')
    print(np.asarray(lista_vel))
    
    print(' ')
    print('************* G vs X *************')   
    print(np.asarray(lista_GX))
    
    print(' ')
    print('************* VEL. mezcla vs X *************')
    print(np.asarray(lista_VhX))
   
    
def lista_puntos(rho_l, rho_g, lim):
    cantidad_puntos = 5
    lista_vel = []
    lista_GX = []
    lista_VhX = []
    
    

    for i in range(cantidad_puntos):
        
        #Variables carta vel sup gas vs vel sup liq
        promediox = 1
        promedioy = 0.1
        varianza = 4
        j_g = np.random.lognormal(promediox, varianza)
        j_l = np.random.lognormal(promedioy, varianza)
       
        while(j_g < lim[0] or j_g > lim[1]):
            j_g = np.random.lognormal(promediox, varianza)
        while(j_l < lim[2] or j_l > lim[3]):
            j_l = np.random.lognormal(promedioy, varianza)           
       
        lista_vel.append([j_g, j_l])

        G, X = transformacion(j_l, j_g, rho_l, rho_g)  
        
        lista_GX.append([X, G])

        lista_VhX.append([X, j_l + j_g])

        
    return lista_vel, lista_GX, lista_VhX
    

def trans_1(x, y, lim):
    

    #Lo pasamos a exponencial para que cuando se cree la escala logaritmica
    #la forma quede igual
    m = 125.
    x = np.power(10, x/m)
    y = np.power(10, y/m)
        
    
    #cambiamos la escala en x
    a = min(x)
    b = max(x)
    c = lim[0]
    d = lim[1]
    x = (x-a)*(d-c)/(b-a) + c

    
    #cambiamos la escala en y
    a = min(y)
    b = max(y)
    c = lim[2]
    d = lim[3]
    y = (y-a)*(d-c)/(b-a) + c
   

    return x,y
   


    
def Mandhane(w_l, w_g, x, A, rho_l, rho_g, lista_velocidades, lim, original=True, velocidad_mezcla=False):
     #Carta de Mandhane
     j_l = w_l*(1-x)/(A*rho_l)
     j_g = w_g*x/(A*rho_g)
     
     _dir = 'figs/md2'
     title = 'Carta de Mandhane'
     
     if(original):
         xlabel = r"$J_G$ (m/s)"
         ylabel = r"$J_L$ (m/s)"
         etiquetas = [
         [2e-1, 6, 'Burbuja'],
         [1e-1, 1, 'Alternante'],
         [2, 1, 'Pulsante'],
         [1e-1, 3e-2, 'Estratificado'],
         [3.5, 5e-2, 'Ondulado'],
         [2e1, 2e-1, 'Anular Disperso']
          ]
     else:
         if(velocidad_mezcla):
             ylabel = r"$V_h$ (m s$^{-1}$)"
             xlabel = r"x (%)"
             etiquetas = [
             [0.0026, 28, 'Burbuja'],
             [0.0026, 1, 'Alternante'],
             [0.29, 5, 'Pulsante'],
             [6, 0.5, 'Estratificado'],
             [13, 9, 'Ondulado'],
             [6, 57, 'Anular Disperso']
              ]
             
         else:
             ylabel = r"$G$ (Kg m$^{-2}$ s$^{-1}$)"
             xlabel = r"x (%)"
             etiquetas = [
             [0.007, 8000, 'Burbuja'],
             [0.24, 900, 'Pulsante'],
             [0.007, 900, 'Alternante'],
             [0.007, 60, 'Estratificado'],
             [10, 57, 'Ondulado'],
             [7, 2800, 'Anular Disperso']
              ]
        

     
     Curvas(_dir, j_l, j_g, title, xlabel, ylabel, lim, etiquetas, rho_l, rho_g, lista_velocidades, original, velocidad_mezcla)
     
     

     
     
def transformacion(j_l, j_g, rho_l, rho_g):

    G = j_l*rho_l +  j_g*rho_g 
    
    X = j_g*rho_g/G
    
    #La calidad de retorna como porcentaje
    return G, 100*X


    
def Curvas(_dir, xp, yp, title, xlabel, ylabel, lim, etiquetas, rho_l, rho_g, lista_velocidades, original=True, velocidad_mezcla=False):
    
    #Definimos el lienzo de la grafica
    fig = plt.figure()
    plt.title(title)
    ax1 = plt.gca()
    img = Image.open(_dir + '.png')
    
    #Se convierte a array, pero teniendo en cuenta que los .png cuentan desde 
    # abajo (por eso el [:,::-1]) y los ejes 'x' y 'y' estan invertidos (por 
    # eso el .T)
    arr = np.array(img.convert('L')).T[:,::-1]
    
    #Se filtra la imagen, se dejan solo los pixeles negros (brillo menor a 100)
    indices = np.argwhere(arr < 100)
    
    #Se filtran pixeles usando un promedio pesado
    points, labels = mean_shift(indices, bandwidth=1.0)

    #Componentes en x y 'y' de los pixeles
    x = points[:,0]
    y = points[:,1] 
    
    #Se hace la transformacion de pixeles a j_l (velocidad superficial del liquido)
    # y j_g (lo mismo pero en el gas)
    j_g, j_l = trans_1(x, y, lim)
    
    


    if(original):
        ax1.scatter(j_g,j_l, s=1, c='k')
        
        
    else: 
        
        #Se hace la transformacion de coordenadas a G (flujo volumetrico) y X (calidad)
        G, X = transformacion(j_l, j_g, rho_l, rho_g)
        ax1.set_xlim(min(X), max(X))
        if(velocidad_mezcla):
            v_h = j_g + j_l
            ax1.scatter(X,v_h, s=1, c='k')
        else:
            ax1.scatter(X,G, s=1, c='k')
    

    
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    #Graficar las etiquetas de las regiones
    for i in range(len(etiquetas)):
        ax1.text(etiquetas[i][0], etiquetas[i][1], etiquetas[i][2])

    #Graficar puntos
    for i in range(len(lista_velocidades)):
        ax1.axvline(lista_velocidades[i][0], c='g', ls='-', alpha=0.5, lw=1)
        ax1.axhline(lista_velocidades[i][1], c='g', ls='-', alpha=0.5, lw=1)
        ax1.text(lista_velocidades[i][0]*1.1, lista_velocidades[i][1]*1.1, str(i+1))
        ax1.plot(lista_velocidades[i][0], lista_velocidades[i][1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor='b')
        

    plt.show()    


    
    
    
if 'name'== __main__():
    __main__()