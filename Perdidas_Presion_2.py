#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:19:44 2022

@author: fluidos
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fsolve

pi = np.pi
power = np.power
arcos = np.arccos
sqrt = np.sqrt

def main():
    
    
    '''--------------------------------*
    | Parametros de entrada del modelo |
    | de Dukler y Hubbard (1975)       |
    *--------------------------------'''
    
 
    #densidad del liquido en kg/m^3
    rho_l = 1000
 
    #densidad del gas en kg/m^3
    rho_g = 1.6
    
    #viscosidad del liquido en Pa s
    mu_l = 1e-3
 
    #viscosidad del gas en Pa s
    mu_g = 0.02e-3
 
    #Angulo de inclinacion en grados (el modelo es solo 
    #para tuberias horizontales o ligeramente inclinadas)
    theta = 0*pi/180.
 
    #Flujo masico del liquido en kg/s
    w_l = 1
 
    #Flujo masico del gas en kg/s
    w_g = 0.01135  
    
    #diámetro de la tuberia en m
    d = 2.53*1e-2
    
    #Temperatura del gas en *C
    T = 20 + 273.15

    #Presion del gas en Pa
    p_g = 1
    
    #Volumen del gas en m^3
    V_g = 1
    
     
    #Gravedad en m/s**2
    g = 9.81
    
    
    '''-------------------------------*
    | Parametros derivados del modelo |
    | de Dukler y Hubbard (1975)      |
    *-------------------------------'''
    
    #Area de la tuberia en m^2
    A = 0.25*pi*d*d
    
    #Flujo masico del liquido en m^3/s
    q_l = w_l/rho_l
 
    #Flujo masico del gas en m^3/s
    q_g = w_g/rho_g
    
    #Velocidad superficial del liquido en m/s
    V_sl = q_l/A 

    #Velocidad superficial del gas en m/s
    V_sg = q_g/A 
    
    
    #print(A, q_l, q_g, V_sl, V_sg)
    
    
    #Velocidad del slug/mezcla en m/s
    V_s = V_sl + V_sg
   

    #Fracción volumétrica del liquido en el slug (adimensional)
    #usando la correlacion de Gregory Nicholson y Aziz
    H_lls = 1./(1. + (V_s/8.66)**1.39)
    
    #Numero de Reynolds del slug (adimensional)
    Re_s = d*V_s*(rho_l*H_lls + rho_l*(1. - H_lls))/(mu_l*H_lls + mu_l*(1. - H_lls))    
  
    #Constante c (no se que significa) (adimensional)
    c = 0.021*np.log(Re_s) + 0.022
    
    #Velocidad de la burbuja de Taylor en m/s
    V_tb = (1. + c)*V_s
    
    

    #Factor de friccion de Hall-Moody-Fanning-Blasius
    f_t = factor_friccion(d, Re_s)

    #Frecuencia del slug en Hz
    f_s = correlacion_fs(d, V_sl, V_sg, g)
    
    #print(V_s, H_lls, f_s, Re_s, c, V_tb)
    

    '''-------------------------------*
    | Parametros de salida del modelo |
    | de Dukler y Hubbard (1975)      |
    *-------------------------------'''

    
    H_ltbe, l_s, l_f, error_min = Iteraciones(H_lls, d, V_s, g, theta, c, f_t, V_sl, V_tb, f_s, False)
    #print(H_ltbe, l_s, l_f, error_min)
    
    
    #longitud de la unidad
    l_u = l_s + l_f
 
    #Tasa de recogimiento/derramamiento/levantamiento en kg/s
    x = (V_tb - V_s)*rho_l*A*H_lls
    
    #Velocidad del liquido en el slug de equilibrio
    v_ltbe = V_tb - x/(rho_l*A*H_ltbe)
    
    
    #longitud de zona de mezclado
    l_m = 0.3*(V_s - v_ltbe)**2/(2*g)
    
    #velocidad del gas en la burbuja de Taylor minima
    v_gtb_min = V_tb - c*V_s*(1 - H_lls)/(1 - H_lls)

    #velocidad del gas en la burbuja de Taylor maxima
    v_gtb_max = V_tb - c*V_s*(1 - H_lls)/(1 - H_ltbe)
 
    #velocidad del liquido en la burbuja de Taylor
    v_ltb = V_tb - x/(rho_l*A*H_lls)


    
    #Caida de presión por friccion
    P_f = 2*f_t*V_s**2*l_s/d*(rho_l*H_lls + rho_g*(1 - H_lls))
    
    #Caida de presion por aceleracion
    P_a = x/A*(V_s - v_ltbe)
    
    #Caida de presion total
    P = P_a + P_f
    
    '''Calculos necesarios para el Diametro hidralico del liquido en la zona de 
     equilibrio de la burbuja de Taylor
    '''
    
    #Variable Auxiliar
    _2h1 = 2.*H_ltbe - 1

    # Perimetro gas - pared
    S_g = arcos(_2h1)
    
    #Perimetro liquido - pared
    S_l = (pi - S_g)
    
    
    # Perimetro interfaz
    S_i = sqrt(1. - _2h1*_2h1)

    
    #Area liquido
    A_l = 0.25*(pi - S_g + _2h1*S_i)
    
    
    # Diametro  hidraulico del liquido
    d_l = 4.*A_l/S_l*d
    
    #velocidad superficial del líquido en la zona de equilibrio
    vs_ltbe = v_ltbe*(A_l*d*d)/A

    
    #Numero de reynolds del liquido en la zona de equilibrio de la burbuja de
    #Taylor
    Re_tbe =  v_sltbe*d_l*rho_l/mu_l
    
    
    print(l_s, l_f)
    print(Re_tbe, vs_ltbe)
    print(l_u, x, v_ltbe, l_m)
    print(v_gtb_min, v_gtb_max, v_ltb)
    print(P_f, P_a, P)
  
    
def correlacion_ls(d, lim = 1):
    '''correlación para la longitud del Slug (inicial), muchos de estos estimados
    estan en pulgadas, así que pasamos el diametro a pulgadas y establecemos
    el limite inferior en 1 pulgada
    
    '''
    
    #Pasamos el diametro a pulgadas (factor 39.3701 pulgada/metro)
    d = 39.3701*d

    if(d < lim):
        #Si la tuberia es muy pequeña,  la relacion es sencilla
        ls = 30.*d
        
        #esto da un resultado en pulgadas, hay que pasarlo a metros 
        #(factor 1/39.3701 metro/pulgada)
        return ls/39.3701
    else:
        #Si no, se usa la correlacion usada en el campo de Prudhoe Bay
        ls = np.exp(-2.099 + 4.859*np.sqrt(np.log(d)))
        
        #esto da un resultado en pies, hay que pasarlo a metros 
        #(factor 0.3048 metro/pie)
        return 0.3048*ls
              
    
   
def factor_friccion(d, Re_s):
    #Factor de friccion de Hall-Moody-Fanning-Blasius
    
    #Rugosidad de la tuberia en mm
    e = 0   
    
    return 0.001375*(1. + power(2e4*e/d + 1e6/Re_s, 1./3.))
     

def integral(H_ltb, H_lls, d, V_s, g, theta, c, f_t):


    sin_a = 2.*np.sqrt(H_ltb*(1. - H_ltb))

    cos_a = 1. - 2.*H_ltb
        
    #Angulo de apertura extendido desde el nivel de liquido enla burbuja de 
    #Taylor (H_ltb), este es \theta en el articulo de Dukler-Hubberts,
    alfa = 2.*np.arccos(cos_a)
    
    #Constante B
    B = 1. + c*(1. - H_lls/H_ltb)
    
    #Numero de Froude
    F_r = V_s**2/(g*d)

    '''
    Se dividiará el integrando W en terminos mas sencillos
    
          C_1 -  _1_ (_C_2_ - C_4)
    W = _______F_r_(   C_3     )_____
                    C_5 
    '''

    C_1 = (c*H_lls/H_ltb)**2
    
    C_2 = 0.5*pi*H_ltb*sin_a + sin_a**2

    C_3 = 1. - cos_a
    
    C_4 = 0.5*cos_a
    
    C_5 = f_t*(B)**2*alfa/(pi) + H_ltb/F_r*np.sin(theta)


    W =   (C_1 - (C_2/C_3 - C_4)/F_r)/C_5

    
    return W


def Iteraciones(H_lls, d, V_s, g, theta, c, f_t, V_sl, V_tb, f_s, imprimir = True):
    
    

    #Longitud del slug
    l_s_maximo = correlacion_ls(d)
    
    
    
    l_s = np.linspace(1e-4, l_s_maximo, 5000)    

    #Longitud de la capa (film) de liquido que esta entre la burbuja de Taylor
    #y la tuberia
    l_f = V_tb/f_s - l_s
   
    #Se filtran solo para valores que den longitudes positivas
    l_s = l_s[l_f > 0]
    l_f = l_f[l_f > 0]

    
    #fracción volumetrica de la burbuja de Taylor en equilibrio (inicial)
    H_ltbe = (V_sl + H_lls*(c*V_s - f_s*l_s))/(V_tb - f_s*l_s)  
   

    #Se filtran solo aquellos valores de la fracción volumétrica que sean validos,
    #para ellos se muestran los indices de las entradas positivas y menores que uno
    cond = (H_ltbe > 0) & (H_ltbe < 1)

    
    #Ahora si, se filtran usando las fracciones volumetricas de equilibrio positivas
    H_ltbe = H_ltbe[cond]
    l_f = l_f[cond]
    l_s = l_s[cond]



    error = []
    lf_lista = []
    h_lista = []
    ls_lista = []


    
    #l_f_esperado =  integral(error + l_f, H_lls, d, V_s, g, theta, c, f_t)
    for i in range(len(H_ltbe)):
        l_f_esperado = d*integrate.quad(integral, H_ltbe[i], H_lls, args=(H_lls, d, V_s, g, theta, c, f_t))[0]
        
        #Si la longitud esperada es negativa, hay que filtrar este resultado
        #como invalido, y para conservar los indices, se crean nuevas listas
        #que guardan los valores validos de cada variable
        if(l_f_esperado > 0):
            ls_lista.append(l_s[i])
            error.append(100*abs(l_f[i] - l_f_esperado)/l_f[i])
            h_lista.append(H_ltbe[i])
            lf_lista.append(l_f[i])

    #Se obtiene el indice del error mas bajo
    indice = np.where(error == np.min(error))[0][0]
 

    if(imprimir):   
        graficar_error(ls_lista, error)
    #graficar_funcion(H_lls, d, V_s, g, theta, c, f_t, H_ltbe[indice])
   
    #como las listas nuevas (error, lf_lista, h_lista y ls_lista ) se crearon
    #en la misma condición, tienen la misma cantidad de elementos y los indices
    #se preservan
    he_valida = h_lista[indice]
    l_s_valido = ls_lista[indice]
    l_f_valido = lf_lista[indice]
    error_min = error[indice]

    return he_valida, l_s_valido, l_f_valido, error_min


def graficar_funcion(H_lls, d, V_s, g, theta, c, f_t, H_ltbe):

    
    #Array de fracción volumetrica de la burbuja de Taylor
    H_ltb = np.linspace(H_ltbe, H_lls, 150)

    res = np.zeros(H_ltb.shape)
    
    for i in range(len(H_ltb)):
        res[i] = integral(H_ltb[i], H_lls, d, V_s, g, theta, c, f_t)
    
    plt.title(r'Integrando $W(H_{ltb})$')
    plt.xlabel(r'$H_{ltb}$ (*)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel(r'$W(H_{ltb})$ (*)')
    plt.ylim(bottom=0, top=max(res))   
    
    plt.plot(H_ltb, res)
    plt.show()
    

def graficar_error(H_ltbe, error):


    plt.title(r'Valor absoluto del Error porcentual ')
    plt.xlabel(r'$l_{s}$ (m)')
    plt.ylabel(r'| $E_{rel}$ | (%)')
    plt.ylim(bottom=0, top=100)     
    
    plt.plot(H_ltbe, error)
    plt.show()
    
    
def correlacion_fs(d, V_sl, V_sg, g):   
    #Correlación de Gregory y Scott 1969
    v_ns = 1.25*(V_sg + V_sl)
    
    fs_corre = 0.0226*(V_sl/(g*d)*(19.75/v_ns + v_ns))**(1.2)
    
    return fs_corre
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()