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


def main():
    
    
    '''--------------------------------*
    | Parametros de entrada del modelo |
    | de Dukler y Hubbard (1975)       |
    *--------------------------------'''
    
 
    #densidad del liquido en kg/m^3
    rho_l = 1000
 
    #densidad del gas en kg/m^3
    rho_g = 1.6
    
    #viscosidad del liquido en kg/m^3 (Pa s)
    mu_l = 1e-3
 
    #viscosidad del gas en kg/m^3 (Pa s)
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
    
    #Frecuencia del slug en Hz
    f_s = 1
     
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
    
    #Tasa de recogimiento/derramamiento/levantamiento en kg/s
    x = (V_tb - V_s)*rho_l*A*H_lls

    #Factor de friccion de Hall-Moody-Fanning-Blasius
    f_t = factor_friccion(d, Re_s)



    Iteraciones(H_lls, d, V_s, g, theta, c, f_t, V_sl, V_tb, f_s)
   

    
  
    
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
    B = 1. + c*(1. - H_ltb/H_lls)
    
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
    
    C_5 = f_t*(B*V_s)**2*alfa/(pi*d) + H_ltb/F_r*np.sin(theta)


    W =   (C_1 - (C_2/C_3 - C_4)/F_r)/C_5

    
    return W


def Iteraciones(H_lls, d, V_s, g, theta, c, f_t, V_sl, V_tb, f_s):
    
    

    def evaluar(H_ltbe, H_lls, d, V_s, g, theta, c, f_t, l_f):
        if(H_ltbe >= 1):
            H_ltbe = 0.999
        if(H_ltbe <= 0):
            H_ltbe = 0.001
        #Realizamos la integracion numerica      
        return d*integrate.quad(integral, H_ltbe, H_lls, args=(H_lls, d, V_s, g, theta, c, f_t))[0] - l_f

    #Longitud del slug
    l_s = correlacion_ls(d)

    #Longitud de la capa (film) de liquido que esta entre la burbuja de Taylor
    #y la tuberia
    l_f = V_tb/f_s - l_s
   

    l_f_esperado = 0
    
    #fracción volumetrica de la burbuja de Taylor en equilibrio (inicial)
    H_ltbe = (V_sl - H_lls*(c*V_s + f_s*l_s))/(V_tb - f_s*l_s)  
        
    
    #Si los puntos a y b encuentran un
    cosa = fsolve(evaluar, H_ltbe, args=(H_lls, d, V_s, g, theta, c, f_t, l_f))[0]

    #l_f_esperado = d*integrate.quad(integral, cosa, H_lls, args=(H_lls, d, V_s, g, theta, c, f_t))[0]
    #print(H_ltbe, H_lls, l_f, l_s)
    print(cosa, l_f_esperado, l_f, H_ltbe)
    
    
    



    graficar_funcion(H_lls, d, V_s, g, theta, c, f_t, H_ltbe)



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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()