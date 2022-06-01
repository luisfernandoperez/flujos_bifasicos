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
    
    #Velocidad superficial del liquido
    Vs_l = w_l/(rho_l*A)
    
    #Velocidad superficial del liquido
    Vs_g = w_g/(rho_g*A)

    #Calidad
    x = w_g/(w_g + w_l)
    
    #flujo masico de la mezcla
    G = (w_l + w_g)/A


    #Mandhane_Original(w_l, w_g, x, A, rho_l, rho_g)
    Mandhane_Transformado(w_l, w_g, x, A, rho_l, rho_g)

    
def Mandhane_Original(w_l, w_g, x, A, rho_l, rho_g):
    #Carta de Mandhane
    j_l = w_l*(1-x)/(A*rho_l)
    j_g = w_g*x/(A*rho_g)
    
    _dir = 'figs/md'
    title = 'Carta de Mandhane'
    xlabel = r"$J_G$ (m/s)"
    ylabel = r"$J_L$ (m/s)"
    lim = [1e-2, 1e2, 1e-3, 1e1]
    etiquetas = [
    [2e-1, 6, 'Burbuja'],
    [1e-1, 1, 'Plug'],
    [2, 1, 'Slug'],
    [1e-1, 3e-2, 'Estratificado'],
    [3.5, 5e-2, 'Ondulado'],
    [2e1, 2e-1, 'Anular']
     ]
    
    Curvas_Originales(_dir, j_l, j_g, title, xlabel, ylabel, lim, etiquetas)
    
    
    print(' ')
    print('************* CARTA DE MANDHANE *************')
    print('* Velocidad Superficial liquido: {:.2e} m/s'.format(j_l))
    print('* Velocidad Superficial gas: {:.2e} m/s'.format(j_g))
 


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
   



def Curvas_Originales(_dir, xp, yp, title, xlabel, ylabel, lim, etiquetas):
    
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

    #Graficamos!
    ax1.scatter(j_g,j_l, s=1, c='k')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(xlabel)
    ax1.set_xlabel(ylabel)
    ax1.set_xlim(lim[0], lim[1])
    ax1.set_ylim(lim[2], lim[3])


    #Identificamos las regiones
    for i in range(len(etiquetas)):
        ax1.text(etiquetas[i][0], etiquetas[i][1], etiquetas[i][2])

    """
    #Graficamos el punto de prueba (esto era en la tarea 2, aca no hace falta)
    ax1.axvline(xp, c='b')
    ax1.axhline(yp, c='b')
    ax1.plot(xp, yp, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    """
    
    
    plt.show() 


    
def Mandhane_Transformado(w_l, w_g, x, A, rho_l, rho_g):
     #Carta de Mandhane
     j_l = w_l*(1-x)/(A*rho_l)
     j_g = w_g*x/(A*rho_g)
     
     _dir = 'figs/md2'
     title = 'Carta de Mandhane'
     xlabel = r"$G$ (Kg m$^{-2}$ s$^{-1}$)"
     ylabel = r"x (%)"
     lim = [1e-2, 1e2, 1e-3, 1e1]
     lim1 = [0.01, 100]
     etiquetas = [
     [0.007, 8000, 'Burbuja'],
     [0.24, 900, 'Plug'],
     [0.007, 900, 'Slug'],
     [0.007, 60, 'Estratificado'],
     [10, 57, 'Ondulado'],
     [7, 2800, 'Anular']
      ]
     
     Curvas_Transformadas(_dir, j_l, j_g, title, xlabel, ylabel, lim, lim1, etiquetas, rho_l, rho_g)
     
     
     print(' ')
     print('************* CARTA DE MANDHANE *************')
     print('* Velocidad Superficial liquido: {:.2e} m/s'.format(j_l))
     print('* Velocidad Superficial gas: {:.2e} m/s'.format(j_g))
     
     
def transformacion(j_l, j_g, rho_l, rho_g):

    G = j_l*rho_l +  j_g*rho_g 
    
    X = j_g*rho_g/G
    
    #La calidad de retorna como porcentaje
    return G, 100*X


    
def Curvas_Transformadas(_dir, xp, yp, title, xlabel, ylabel, lim, lim1, etiquetas, rho_l, rho_g):
    
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
    
    #Se hace la transformacion de coordenadas a G (flujo volumetrico) y X (calidad)
    G, X = transformacion(j_l, j_g, rho_l, rho_g)

    
    ax1.scatter(X,G, s=1, c='k')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(min(X), lim1[1])
    ax1.set_ylabel(xlabel)
    ax1.set_xlabel(ylabel)


    for i in range(len(etiquetas)):
        ax1.text(etiquetas[i][0], etiquetas[i][1], etiquetas[i][2])


    #yp, xp =  transformacion(xp, yp, rho_l, rho_g)
    #ax2.plot(xp, yp, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")

    plt.show()    

    
    
    
if 'name'== __main__():
    __main__()