#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:26:13 2022

@author: lffp
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
sys.path.append(os.path.abspath("/Users/lffp/Documents/posgrado/materia_bifasico/4 Modelado de Transiciones"))
from Modelado_Transiciones import Parametros as t_Ducklker

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
    
    Parametros(d, rho_l, w_l, mu_l, rho_g, w_g, mu_g, T, theta)

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
    

    '''---------------------*
    | CALCULO DE GRADIENTES |
    *---------------------'''
    #grad_fricc, grad_grav, grad_acc = gradientes(d, theta, g, w_l, w_g, A, T, rho_l, rho_g, mu_l, mu_g, x, G, Vs_l, Vs_g)

    
    '''---------------------*
    | MODELO HEWITT-ROBERTS |
    *---------------------'''
    #Hewitt_Roberts(G, x, rho_l, rho_g)
    

    '''---------------------------*
    | SUPONIENDO FLUJO HORIZONTAL |
    *---------------------------'''
    #En el flujo horizontal, el angulo es cero
    theta = 0.
    
    #Carta de Mandhane
    #Mandhane(w_l, w_g, x, A, rho_l, rho_g)
    
    #Modelo de Taitel - Dukler
    #Taitel_Dukler(d, rho_l, w_l, mu_l, rho_g, w_g, mu_g, theta)

    #Modelo homogeneo de Wallis
    #Caida_Presion_homogeneo(d, theta, g, w_l, w_g, A, T, rho_l, rho_g, mu_l, mu_g, x, G, Vs_l, Vs_g)

    #Modelo de Lockhart y Martinelli
    #Lockhart_Martinelli(d, rho_l, Vs_l, mu_l, rho_g, Vs_g, mu_g, w_l, w_g, x, A,)
    
    #Modelo de Dukler Eaton y Flanigan
    #Dukler_Eaton(Vs_g, Vs_l, rho_l, rho_g, mu_l, mu_g, d, g)

    #Modelo Beggs y Brills
    #Beggs_Brills(Vs_g, Vs_l, rho_l, rho_g, mu_l, mu_g, d, g, theta)


def Beggs_Brills(Vs_g, Vs_l, rho_l, rho_g, mu_l, mu_g, d, g, theta):
    
    #Tension Superficial entre liquido y gas
    sigma = 1.
    
    #Presion inicial desconocida
    p  = 10.
    
    #Rougness relativo del pvc
    e = 0.0015
    
    
    V_m = Vs_l + Vs_g
    
    lambda_l = Vs_l/V_m

    F_r = V_m**2/(d*g)

    patron, A = Patrones_Beggs_Brills(lambda_l, F_r)
    
    
    H_l = Hold_Up(patron, lambda_l, F_r, A, theta, Vs_l, rho_l, g, sigma)
    
    
    grad_fricc = grad_fricc_beggs(V_m, d, lambda_l, H_l, rho_l, rho_g, mu_l, mu_g, e)
    
    rho_s = rho_l*H_l + rho_g*(1. - H_l)
    
    
    grad_grav = rho_s*g*np.sin(theta)
    
    
    E_k = rho_s*V_m*Vs_g/p
    
    
    grad_presion = (grad_fricc + grad_grav)/(1. - E_k)
    

    print(' ')
    print('****** CAIDA PRESION BEGGS BRILSS ********')
    print('* Caida de Presión: {:.2f} Pa/m'.format(grad_presion))
    
    


def grad_fricc_beggs(V_m, d, lambda_l, H_l, rho_l, rho_g, mu_l, mu_g, e):
    
    
    rho_n = rho_l*lambda_l + rho_g*(1. - lambda_l)
    mu_n = mu_l*lambda_l + mu_g*(1. - lambda_l)
    
    Re_n = rho_n*V_m*d/mu_n
    
    y = lambda_l/H_l**2
    
    s = 0
    
    
    if (y > 1. and y < 1.2):
        s = np.log(2.2*y-1.2)
    else:
        x = np.log(y)
        s = x/(-0.0523 + 3.182*x - 0.8725*x**2 + 0.01853*x**4)
        

    f_n = 0.0055*(1 + power(2e4*e/d + 1e6/Re_n, 1./3.))
    
    f = f_n*np.exp(s)
        
    return f*rho_n*V_m**2/(2.*d)







def Patrones_Beggs_Brills(lambda_l, F_r):
    
    L_1 = 316*power(lambda_l, 0.302)

    L_2 = 0.1*power(lambda_l,-1.4516)
    
    L_3 = 0.0009252*power(lambda_l, -2.4684)
    
    L_4 = 0.5*power(lambda_l, -6.738)


    Segregado = (lambda_l < 0.01 and F_r < L_1) or (lambda_l >= 0.01 and F_r < L_2)
    Transicion = lambda_l >= 0.01 and F_r <= L_3 and F_r >= L_2
    Intermitente = (lambda_l >= 0.01 and lambda_l <= 0.4 and F_r <= L_1 and F_r > L_3) or (lambda_l >= 0.4 and F_r <= L_4 and F_r >= L_3)
    Distribuido = (lambda_l < 0.4 and F_r >= L_1) or (lambda_l >= 0.4 and F_r > L_4)
    
    respuesta = ''
    
    if(Segregado):
        respuesta = 'Segregado'
    elif(Transicion):
        respuesta = 'Transicion'
    elif(Intermitente):
        respuesta = 'Intermitente'
    elif(Distribuido):
        respuesta = 'Distribuido'
        
        
    A = (L_3 - F_r)/(L_3 - L_2)
        
    return respuesta, A
    


def Psi_Patron_Flujo(patron, lambda_l, theta, F_r, N_lv):
    
    d = None
    c = None
    f = None
    g = None
    C = None
    
    if(theta <= 0):
        d = 4.7
        c = -0.3692
        f = 0.1244
        g = -0.5056
    else:
        if(patron == 'Segregado'):
            d = 0.011
            c = -3.768
            f = 3.539
            g = -1.614
        elif(patron == 'Intermitente'):
            d = 2.69
            c = 0.305
            f = -0.4473
            g = 0.0978
            
    
    if(patron == 'Distribuido' and theta > 0):
        Psi = 1.
    else:
        C = (1 - lambda_l)*np.log(d*power(lambda_l,c)*power(N_lv, f)*power(F_r,g))
        Psi = 1. + C*(np.sin(1.8*theta) - 0.333*(np.sin(1.8*theta))**3)
        
    return Psi



def N_vl(Vs_l, rho_l, g, sigma):
    #revisar unidades!
    n_vl = 1.938*Vs_l*power(rho_l/(g*sigma),0.25)
    return n_vl


def Hold_Up(patron, lambda_l, F_r, A, theta, Vs_l, rho_l, g, sigma ):
    
    
    a = None
    b = None
    c = None
    H_l = None
    Psi = None
    n_vl = N_vl(Vs_l, rho_l, g, sigma)
    
    
    if(patron == 'Segregado'):
        a = 0.98
        b = 0.4846
        c = 0.0868
    elif(patron == 'Intermitente'):
        a = 0.845
        b = 0.5351
        c = 0.0173
    elif(patron == 'Distribuido'):
        a = 1.065
        b = 0.5824
        c = 0.0609
        

    if(patron == 'Transicion'):
        _as = 0.98
        bs = 0.4846
        cs = 0.0868
        ai = 0.845
        bi = 0.5351
        ci = 0.0173
        
        H_l0s = _as*power(lambda_l,bs)/power(F_r,cs)
        H_l0i = ai*power(lambda_l,bi)/power(F_r,ci)
        
        Psis = Psi_Patron_Flujo('Segregado', lambda_l, theta, F_r, n_vl)
        Psii = Psi_Patron_Flujo('Intermitente', lambda_l, theta, F_r, n_vl)
        
        H_l = A*H_l0s*Psis + (1. - A)*Psis*Psii
        
    else:
        H_l0 = a*power(lambda_l,b)/power(F_r,c)
        Psi = Psi_Patron_Flujo(patron, lambda_l, theta, F_r, n_vl)
        H_l = H_l0*Psi
    
    if(H_l0 < lambda_l):
        print('El Hold Up es menor que la fraccion volumetrica superficial: revisar')
        exit()
    else:
        return H_l

def Dukler_Eaton(Vs_g, Vs_l, rho_l, rho_g, mu_l, mu_g, d, g):
    

    
    H_l = 1./(1 + 0.3264*power(Vs_g*3.281,1.006))
    
    V_m = Vs_l + Vs_g
    
    lambda_l = Vs_l/V_m
    
    rho_k = rho_l*lambda_l**2/H_l + rho_g*(1.- lambda_l)**2/(1. - H_l)
    
    mu_m = (lambda_l*mu_l + (1- lambda_l)*mu_g)
    
    Re_k = rho_k*V_m*d/mu_m

    f_n = 0.00140 + 0.125*power(Re_k,-0.32)
    
    y =  - np.log(lambda_l)
    
    factor = 1.281 - 0.478*y + 0.444*y**2 - 0.094*y**3 + 0.00843*y**4
    f_tp = (1. + y/factor)*f_n
    
    grad_presion = f_tp*rho_k*V_m**2/(2*d)
    

    print(' ')
    print('****** CAIDA PRESION DUKLER EATON FLANIGAN *****')
    print('* Caida de Presión: {:.2f} Pa/m'.format(grad_presion))




def Lockhart_Martinelli(d, rho_l, Vs_l, mu_l, rho_g, Vs_g, mu_g, w_l, w_g, x, A):

    #Numeros de Reynolds
    Re_l = rho_l*Vs_l*d/mu_l
    Re_g = rho_g*Vs_g*d/mu_g
    
    #Fijamos los parámetros para flujos laminares
    m = 1.
    n = 1.
    C_l = 16.
    C_g = 16.

    #Variables auxiliares de la condicion de cada fase
    cg = 'laminar'
    cl = 'laminar'
    
    #Si alguna fase es turbulenta, cambiamos los parametros para turbulentos
    if(Re_l > 2000):
        n = 0.2
        C_l = 0.046
        cl = 'turbulento'

    if(Re_g > 2000):
        m = 0.2
        C_g = 0.046
        cg = 'turbulento'
        
    #Parámetro de Lockhart y Martinelli
    X = (C_l/C_g)*(Re_g**m/Re_l**n)*(rho_l/rho_g)*(Vs_l/Vs_g)**2
    X = sqrt(X)

    #Parametro de ajuste del metodo de Chisholm
    P_C = 0
    
    #Dependiendo de los flujos, este parametro tiene un valor
    if (cg == 'laminar' and cl == 'laminar'):
        P_C = 5
    elif (cg == 'laminar' and cl == 'turbulento'):
        P_C = 10
    elif (cg == 'turbulento' and cl == 'laminar'):
        P_C = 12
    else:
        P_C = 20

        
    phi_l = 1. + P_C/X + 1./X**2
    

    f_l = C_l*power(Re_l,-n)
    
    grad_presion = phi_l*2.*f_l*(Vs_l*A*rho_l)**2/(d*A**2*rho_l)

    print(' ')
    print('****** CAIDA PRESION LOCKHART Y MARTINELLI *****')
    print('* Caida de Presión: {:.2f} Pa/m'.format(grad_presion))
        
        

def Tabla_a_b(X, cl, cg):
    '''
    Tabla sacada de Chisholm 1967 de la tabla 4 en la pagina 1775
    '''
    a = 0
    b = 0
    
    if (cg == 'laminar' and cl == 'laminar'):
        if(X < 0.1):
            a = 0.071
            b = 3.23
        if(X >= 0.1 and X < 1.):
            a = 0.288
            b = 4.79
        if(X >= 1. and X < 10.):
            a = 0.754
            b = 1.58
        if(X >= 10. and X < 100.):
            a = 0.963
            b = 1.53
            
    elif (cg == 'laminar' and cl == 'turbulento'):
        if(X < 0.1):
            a = 0.057
            b = 7.82
        if(X >= 0.1 and X < 1.):
            a = 0.278
            b = 4.48
        if(X >= 1. and X < 10.):
            a = 0.73
            b = 1.72
        if(X >= 10. and X < 100.):
            a = 0.995
            b = 1.05
            
    elif (cg == 'turbulento' and cl == 'laminar'):
        if(X < 0.1):
            a = 0.077
            b = 2.69
        if(X >= 0.1 and X < 1.):
            a = 0.355
            b = 2.46
        if(X >= 1. and X < 10.):
            a = 0.954
            b = 1.11
        if(X >= 10. and X < 100.):
            a = 1.172
            b = 0.44
            
    else:
        if(X < 0.1):
            a = 0.08
            b = 2.56
        if(X >= 0.1 and X < 1.):
            a = 0.331
            b = 2.53
        if(X >= 1. and X < 10.):
            a = 0.846
            b = 1.26
        if(X >= 10. and X < 100.):
            a = 1.138
            b = 0.48
            

    return a,b


def Caida_Presion_homogeneo(d, theta, g, w_l, w_g, A, T, rho_l, rho_g, mu_l, mu_g, x, G, Vs_l, Vs_g):
    
    grads = gradientes(d, theta, g, w_l, w_g, A, T, rho_l, rho_g, mu_l, mu_g, x, G, Vs_l, Vs_g)
    
    CP_homo = sum(grads)
    print(' ')
    print('************ CAIDA PRESION HOMOGENEO ************')
    print('* Caida de Presión Modelo Homogéneo de Wallis: {:.2f} Pa/m'.format(CP_homo))
    
    
    
def Taitel_Dukler(d, rho_l, w_l, mu_l, rho_g, w_g, mu_g, theta):
    #En la tarea 1 se hizo el modelo de  Taitel - Duckler, pero dependiendo de
    # los caudales en metros cubicos por hora, asi que se tienen que crear estos
    #caudales en estas unidades
    q_l = w_l/rho_l*3600
    q_g = w_g/rho_g*3600

    #El diametro se vuelve a pasar a cm
    d = d*100

    print(' ')
    print('************* MODELO TAITEL Y DUKLER *************')
    t_Ducklker(d, rho_l, q_l, mu_l, rho_g, q_g, mu_g, theta)



def Mandhane(w_l, w_g, x, A, rho_l, rho_g):
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
    
    Curvas(_dir, j_l, j_g, title, xlabel, ylabel, lim, etiquetas)
    
    print(' ')
    print('************* CARTA DE MANDHANE *************')
    print('* Velocidad Superficial liquido: {:.2e} m/s'.format(j_l))
    print('* Velocidad Superficial gas: {:.2e} m/s'.format(j_g))



def Hewitt_Roberts(G, x, rho_l, rho_g):

    #Coordenada horizontal en el diagrama HR
    rho_j_l = (G*(1. - x))**2/rho_l

    #Coordenada vertical en el diagrama HR
    rho_j_g = (G*x)**2/rho_g
    
    _dir = 'figs/hw'
    title = 'Diagrama de Hewitt - Roberts'
    xlabel = r"$\rho_G$ $J_G^2$ (kg/s$^2$ - m)"
    ylabel= r"$\rho_L$ $J_L^2$ (kg/s$^2$ - m)"
    lim = [1e0, 1e5, 1e-1, 1e5]
    etiquetas = [
    [5, 5e3, 'Anular'],
    [6, 9.5, 'Churn'],
    [5e3,5e3, 'Anular Wispy '],
    [50, 0.2, 'Slugs'],
    [1e3, 0.3, 'Burbujeante Slug'],
    [5e3, 20, 'Burbujeante']
     ]
    
    
    
    Curvas(_dir, rho_j_l, rho_j_g, title, xlabel, ylabel, lim, etiquetas)
    
    
    print(' ')
    print('********** DIAGRAMA DE HEWITT ROBERTS **********')
    print('* Coordenada horizontal en el diagrama HR: {:.2e} kg/s^2 - m'.format(rho_j_l))
    print('* Coordenada vertical en el diagrama HR: {:.2e} kg/s^2 - m'.format(rho_j_g))
    print('* Patron estimado: Wipsy Annular')
    


    

def gradientes(d, theta, g, w_l, w_g, A, T, rho_l, rho_g, mu_l, mu_g, x, G, Vs_l, Vs_g):

    #Velocidad de la mezcla
    V_m = Vs_l + Vs_g
    
    #fraccion de volumen superficial del liquido
    lamda_l = Vs_l/V_m
    
    #Densidad de la mezcla
    rho_m = lamda_l*rho_l + (1- lamda_l)*rho_g
    
    #Viscosidad de la mezcla usando semenjanza
    mu_m = lamda_l*mu_l + (1- lamda_l)*mu_g
    
    
    #Numero de Reynolds de la mezcla
    Re = rho_m*V_m*d/mu_m

    #Fijamos los parámetros para flujos laminares
    n = 1.
    C = 16.

    #Variables auxiliares para imprimir el estado de la mezcla
    tf = 'laminar'
    
    #Si alguna fase es turbulenta, cambiamos los parametros para turbulentos
    if(Re > 2000):
        n = 0.2
        C = 0.046
        tf = 'turbulento'


        
    grad_fricc = Friccion(n, C, Re, rho_m, V_m, d)
    
    grad_grav = Gravedad(rho_m, theta, g)
    
    grad_acc = Aceleracion(grad_fricc, grad_grav, w_l, w_g, A, T, rho_g, x, G)


    print(' ')
    print('************* PARAMETROS BASICOS *************')
    print('* Area del tubo: {:.5f} m^2'.format(A))
    print('* Velocidad superficial del liquido: {:.5f} m/s'.format(Vs_l))
    print('* Velocidad superficial del gas: {:.5f} m/s'.format(Vs_g))
    print('* Velocidad superficial de la mezcla: {:.5f} m/s'.format(V_m))
    print('* Fracción volumétrica superficial del liquido: {:.5f}'.format(lamda_l))
    print('* Densidad de la mezcla: {:.5f} kg/m^3'.format(rho_m))
    print('* Viscosidad de la mezcla: {:.5f} kg/(m s)'.format(mu_m))
    print('* Numero de Reynolds de la mezcla: {:.5f} y es un flujo {:s}'.format(Re, tf))
    print(' ')
    print('************* CALCULO DE GRADIENTES *************')
    print('* Gradiente de presion por friccion: {:.2f} Pa/m'.format(grad_fricc))
    print('* Gradiente de presion por gravedad: {:.2f} Pa/m'.format(grad_grav))
    print('* Gradiente de presion por aceleración: {:.2f} Pa/m'.format(grad_acc))
    
    return grad_fricc, grad_grav, grad_acc


def Curvas(_dir, xp, yp, title, xlabel, ylabel, lim, etiquetas):
    
    
    fig = plt.figure()
    plt.title(title)
    
    ax1 = plt.gca()
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    img = Image.open(_dir + '.png')
    
    #Se convierte a array, pero teniendo en cuenta que los .png cuentan desde 
    # abajo (por eso el [:,::-1]) y los ejes 'x' y 'y' estan invertidos (por 
    # eso el .T)
    arr = np.array(img.convert('L')).T[:,::-1]
    
    #Se filtra la imagen, se dejan solo los pixeles negros (brillo menor a 100)
    indices = np.argwhere(arr < 100)
    
    #Se filtran pixeles usando un promedio pesado
    points, labels = mean_shift(indices, bandwidth=1.0)

    x = points[:,0]
    y = points[:,1] 
    
    ax1.scatter(x,y, s=1, c='k')
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(min(y), max(y))
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel(xlabel)
    ax2.set_xlabel(ylabel)
    ax2.set_xlim(lim[0], lim[1])
    ax2.set_ylim(lim[2], lim[3])
    
    for i in range(len(etiquetas)):
        ax2.text(etiquetas[i][0], etiquetas[i][1], etiquetas[i][2])

    
    ax2.axvline(xp, c='b')
    ax2.axhline(yp, c='b')
    ax2.plot(xp, yp, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    
    plt.show()

def Friccion(n, C, Re, rho_m, V_m, d):
    """
    menos el Gradiente de presion debido a la friccion
    """
    
    f_F = C*power(Re, -n)
    return 2.*f_F*rho_m*V_m**2/d
    

def Gravedad(rho_m, theta, g):
    """
    menos el Gradiente de presion debido a la gravedad
    """
    
    return rho_m*g*np.sin(theta)



def Aceleracion(grad_friccion, grad_gravedad, w_l, w_g, A, T, rho_g, x, G):
    """
    menos el Gradiente de presion debido a la aceleriacion.
    Suponiendo calidad y area transversal constantes!!
    Tambien se supone que la velocidad del liquido NO depende de la presión, es decir, es incompresible
    
    """

    

    '''
    Constante universal de los gases en J/(mol K) segun el Instituto de Microelectronica de Sevilla
    http://www2.imse-cnm.csic.es/~fiorelli/cd_fisterv1-1/contenido/actividades/act1/act1-a/datos_aire.htm
    y wikipedia
    
    - Notese que este valor es 8314,46261815324 en L⋅Pa/(K⋅mol), pero la
    conversion de litros a metros cubicos es 1 L = 1/1000 m^3, de ahi puede
    surgir el factor de 1000 veces que hay diferencia con respecto al valor
    que tiene el profesor en la lamina 40 de la presentacion.
    '''
    R = 8.31446261815324
    
    #La Masa molar del aire en kg/kmol es 29, o lo que es lo mismo, 29/1000 kg/mol
    M = 29./1000.
    
    #Variacion del volumen especifico con respecto a la presion
    dv_dp = - M/(R*T*rho_g**2)
    
    #Factor de diferencia que hay entre el menos gradiente de presion total y el menos gradiente
    #de presion por la aceleracion
    factor = - G**2*x*dv_dp
    
    
    return factor/(1 - factor)*(grad_friccion + grad_gravedad)
    
    
 
    
    
    
    
    
    
if 'name'== __main__():
    __main__()