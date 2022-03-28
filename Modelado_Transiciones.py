#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:51:41 2022


    Se quiere solucionar y/o graficar la ecuación numero 7 que aparece en el artículo "A Model for Predicting Flow Regime
    Transitions in Horizon and Near Horizontal Gas-liquid Flow" de Y. TAITEL y A. E. DUKLER de 1976 en la página 49. (Todas 
    las variables están adimensionalizadas):
        
        
            X^2 [(___1___)^n * u_l^2 * _S_l_]  -  [(___1___)^m * u_g^2 *(_S_g_ + _S_i_ + _S_i_)] - 4*Y = 0 
                [(u_l*D_l)              A_l ]     [(u_g*D_g)            ( A_g     A_l     A_g )]
    
    Para simplificar su resolución, se reescribirá de la siguiente forma:
    
                                         X^2*F1(h) -  F2(h) - 4*Y = 0                                             A)
    
    Donde:
        
        
    F1(h) = [(___1___)^n * u_l^2 * _S_l_]           F2(h) = [(___1___)^m * u_g^2 *(_S_g_ + _S_i_ + _S_i_)] 
            [(u_l*D_l)              A_l ]                   [(u_g*D_g)            ( A_g     A_l     A_g )]
    
    
    Se usará el método fsolve de scipy.optimize (opcionalmente, se realiza la grafica de h vs X si se quiere realizar la solución
    grafica, para ello, se debe quitar el simbolo de # en la llamda de la funcion Diagrama_Taitel_Dukler())
    
    
    Luego de obtener el valor de h, se calculan las variables adimensionales del sistema, se estudia el equilibrio del sistema
    para ver que transiciones se han realizado para saber en que fase se encuentra.
    
    Es de conocer que esto solo sirve para tuberías rectas horizontales con solo dos fluidos, cuyos flujos estén desarrollados
    (sin efectos de borde o con los efectos al minimo) y si tiene inclinación alguna, esta debe ser muy pequeña, también el
    gradiente de presión del liquido debe ser la misma que la del gas.
    
    
@author: lffp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import cm
import matplotlib.ticker as mticker

#Variables Auxilixares
pi = np.pi
power = np.power
sqrt = np.sqrt
arcos = np.arccos


#Función Principal
def __main__():


    #Parametros del problema
    
    theta = 0.
    d = 5.
    q_l = 0.707
    rho_l = 993
    mu_l = 0.68e-3
    q_g = 21.2
    rho_g = 1.14
    mu_g = 1.98e-5
    
    Parametros(d, rho_l, q_l, mu_l, rho_g, q_g, mu_g, theta)


    #Descomentar solo si se quiere ver la forma del diagrama h vs X
    Diagrama_Taitel_Dukler()




def Parametros(d, rho_l, q_l, mu_l, rho_g, q_g, mu_g, theta=0, g=9.81):
    """
    Entrada donde se introducen los parámetros del problema e imprime en pantalla los valores
    deseados adimensionales

    Parameters
    ----------
    d : float
        Diametro del tubo en cm.
    rho_l : float
        densidad del liquido en kg/m^3.
    q_l : float
        flujo del liquido en m^3/hr.
    mu_l : float
        viscocidad del liquido en kg/m*s.
    rho_g : float
        densidad del gas en kg/m^3.
    q_g : float
        flujo del gas en m^3/hr.
    mu_g : float
        viscocidad del gas en kg/m*s.
    theta : float
        Angulo de inclinación de la tuberia en grados. The default is 0.
    g : TYPE, optional
        modulo de la gravedad en m/s^2. The default is 9.81.
        
    Returns
    -------
    None.

    """
    
    
    #Pasar el diametro a m
    d = d/100.
    
    #Pasar el angulo a radiantes
    theta = theta*pi/180.
    
    #Pasar los flujos a kg/s
    q_l = q_l/3600.
    q_g = q_g/3600.
    
    
    '''----------------------------------*
    | CALCULAR VELOCIDADES SUPERFICIALES |
    *----------------------------------'''
    
    
    #Calculamos el area total
    A = 0.25*pi*d*d
    
    #Velocidades superficiales
    Vs_l = q_l/A
    Vs_g = q_g/A
    
    
    '''--------------*
    | CALCULAR X y Y |
    *--------------'''
    
    #Numeros de Reynolds
    Re_l = rho_l*Vs_l*d/mu_l
    Re_g = rho_g*Vs_g*d/mu_g

    #Fijamos los parámetros para flujos laminares
    m = 1.
    n = 1.
    C_l = 16.
    C_g = 16.

    #Variables auxiliares para imprimir el estado de cada fase
    tfg = 'laminar'
    tfl = 'laminar'
    
    #Si alguna fase es turbulenta, cambiamos los parametros para turbulentos
    if(Re_l > 2000):
        n = 0.2
        C_l = 0.046
        tfl = 'turbulento'

    if(Re_g > 2000):
        m = 0.2
        C_g = 0.046
        tfg = 'turbulento'
        
    
    #Parámetro de Lockhart y Martinelli
    X = (C_l/C_g)*(Re_g**m/Re_l**n)*(rho_l/rho_g)*(Vs_l/Vs_g)**2
    X = sqrt(X)
    
    #Parametro de Inclinacion
    Y = (rho_l - rho_g)*np.sin(theta)*g*power(Re_g, m)*d/(2.*C_g*rho_g*Vs_g*Vs_g)
    
    
    
    '''-----------------------------------------------------*
    | CALCULAR NIVEL EQUILIBRIO Y PARAMETROS ADIMENSIONALES |
    *-----------------------------------------------------'''
    
    
    #Distancia Adimensional inicial, necesaria para el algoritmo
    h_0 = 0.5
    
    #Distancia Adimensional de equilibrio -> ACA SE RESUELVE NUMERICAMENTE LA ECUACION A)
    _h = fsolve(Ecuacion_Taitel_Dukler, h_0, args=(X, Y, m, n))[0]


    #Variable Auxiliar
    _2h1 = 2.*_h-1.

    # Perimetro gas - pared
    S_g = arcos(_2h1)
    
    # Perimetro liquido - pared
    S_l = pi - S_g
    
    # Perimetro interfaz
    S_i = sqrt(1. - _2h1*_2h1)

    #S_i = sqrt(1. - _2h1*_2h1)
    
    #Area liquido
    A_l = 0.25*(pi - S_g + _2h1*S_i)
    
    #Area gas
    A_g = 0.25*(S_g - _2h1*S_i)
    
    #Area total
    A = 0.25*pi
    
    #Velocidad del liquido
    u_l = A/A_l
    
    #Velocidad del gas
    u_g = A/A_g
    
    # Diametro  hidraulico del liquido
    D_l = 4.*A_l/S_l
    
    #Diametro hidraulico del gas
    D_g = 4.*A_g/(S_g + S_i)
    


    '''-------------*
    |  TRANSICIONES |
    *-------------'''


    #Transicion de flujo estratificado a no estratificado
    def transicion_A(_h, C_l, Vs_g, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i, u_g):
        
        # Numero de froude
        F = ( rho_g*Vs_g**2 )/( (rho_l - rho_g)*d*g*np.cos(theta) )
        F = sqrt(F)

        condicion = F**2*( u_g**2*S_i )/( (1 - _h)**2* A_g )

        if condicion >= 1:
            #no estratificado, revisar anular o burbujeante/intermitente en transicion B
            return transicion_B(_h, C_l, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i)   
        
        else:
            #estratificado, revisar liso u ondulado en transicion C
            return transicion_C(F, Re_l, u_l, u_g)


    #Transicion de flujo anular a burbujeante o intermitente
    def transicion_B(_h, C_l, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i):
        
        if _h >= 0.35:
            #Revisar si es burbujetante o intermitente
            return transicion_D(C_l, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i)
        else:
            #Es Anular
            return 'Anular'
    

    # Transicion Flujo Estratificado Liso a Ondulado
    def transicion_C(F, Re_l, u_l, u_g):
        
        #Numero de Duckler
        K = F*sqrt(Re_l)
        
        #Coeficiente de recuperacion
        s = 0.01
        
        condicion = 2/( sqrt(u_l*s) * u_g )
        
        if K >= condicion:
            #Es ondulado
            return 'Estratificado Ondulado'
        else:
            #Es liso
            return 'Estratificado Liso'


    # Transicion Flujo Intermitente a Burbujeante
    def transicion_D(C_l, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i):
        
        #Numero de Taitel
        dpdx_l = 2*C_l*rho_l*Vs_l**2 /( d*Re_l**n)
        T = dpdx_l/((rho_l - rho_g)*np.cos(theta)*g)
        T = sqrt(T)
        
        condicion = ( 8*A_g*(u_l - D_l )**n )/( S_i*u_l**2 )
        condicion = sqrt(condicion)

        if T >= condicion:
            #Es Burbujeante
            return 'Burbujeante'
        else:
            #Es Intermitente
            return 'Intermitente'



    estado = transicion_A(_h, C_l, Vs_g, Vs_l, Re_l, d, rho_l, rho_g, theta, g, u_l, A_g, n, D_l, S_i, u_g)



    '''-------------------------*
    ~  IMPRESION DE RESULTADOS  ~
    *-------------------------'''

    print('* Reynolds del gas: {:.2f}  y es un flujo '.format(Re_g) + tfg )
    print('* Reynolds del liquido: {:.2f}  y es un flujo '.format(Re_l) + tfl )
    print('* Parametro de Lockhart y Martinelli X: {:.5f}'.format(X))
    print('* Parametro de Inclinacion Y: {:.5f}'.format(Y))
    print('* Altura en equilibrio adimensional: {:.3f}'.format(_h))
    print('* Perimetro Gas-Pared adimensional: {:.3f}'.format(S_g))
    print('* Perimetro Liquido-Pared adimensional: {:.3f}'.format(S_l))
    print('* Perimetro Interfaz adimensional: {:.3f}'.format(S_i))
    print('* Area liquido adimensional: {:.3f}'.format(A_l))
    print('* Area gas adimensional: {:.3f}'.format(A_g))
    print('* Velocidad liquido adimensional: {:.3f}'.format(u_l))
    print('* Velocidad gas adimensional: {:.3f}'.format(u_g))
    print('* Diametro hidraulico liquido adimensional: {:.3f}'.format(D_l))
    print('* Diametro hidraulico gas adimensional: {:.3f}'.format(D_g))
    print('* El sistema está en la fase ' + estado)


def Ecuacion_Taitel_Dukler(h, *args):
    """
    Computo de F1(h) y F2(h). Estas a su vez se desglosan en varias variables adimensionales que dependen de h. Comenzamos 
    con las variables mas sencillas de definir con respecto a h.
    

    Parameters
    ----------
    h : float
        Altura adimensional. Varia en el intervalo (tol, 1-tol)
        
    *args -> Contiene:
    X : float
        Parámetro de Lockhart y Martinelli
    Y : float
        Representa las fuerzas relativas actuando en el liquido en la dirección del flujo debido a la gravedad y gradiente de presión
    m : float
        Exponente del factor de fricción para el gas. Para flujos turbulentos m = 0.2 y para laminares m = 1
    n : float
        Exponente del factor de fricción para el liquido. Para flujos turbulentos n = 0.2 y para laminares n = 1

    Returns
    -------
    array de F1(h) y F2(h)

    """
    tol = 1e-5
    
    if h < 0:
        h = tol
    if h > 1:
        h = 1 - tol
        
    #Variable Auxiliar
    _2h1 = 2.*h-1.

    # Perimetro gas - pared
    S_g = arcos(_2h1)
    
    # Perimetro liquido - pared
    S_l = pi - S_g
    
    # Perimetro interfaz
    S_i = sqrt(1. - _2h1*_2h1)

    #S_i = sqrt(1. - _2h1*_2h1)
    
    #Area liquido
    A_l = 0.25*(pi - S_g + _2h1*S_i)
    
    #Area gas
    A_g = 0.25*(S_g - _2h1*S_i)
    
    #Area total
    A = 0.25*pi
    
    #Velocidad del liquido
    u_l = A/A_l
    
    #Velocidad del gas
    u_g = A/A_g
    
    # Diametro  hidraulico del liquido
    D_l = 4.*A_l/S_l
    
    #Diametro hidraulico del gas
    D_g = 4.*A_g/(S_g + S_i)
    
    #Parametros necesarios en la ecuacion
    X = args[0]
    Y = args[1]
    m = args[2]
    n = args[3]
    
    #Funciones F1(h) y F2(h)
    f1_h = power(u_l*D_l, -n)*u_l*u_l*S_l/A_l
    f2_h = power(u_g*D_g, -m)*u_g*u_g*(S_g/A_g + S_i/A_l + S_i/A_g)
    
    return X**2*f1_h - f2_h + 4.*Y


def Diagrama_Taitel_Dukler(puntos=1000, tol = 0.001):
    """
     Acá se realiza la grafica de h vs X realizada por Taitel y Dukler, pero en vez de ser
     h = h(X, Y), la cual es una función no invertible analiticamente, se grafica X = X(h,Y),
     la cual se puede despejar para un h dado.

    Parameters
    ----------
    puntos : integer, optional
        Cantidad de puntos para la resolucion en h. The default is 50.
    tol : float, optional
        Tolerancia de h para evitar problemas de presición en los limites en el rango (0, 1). The default is 1e-1.


    Returns
    -------
    None.


    """

    def Funciones_h(h, m, n):
        """
        Computo de F1(h) y F2(h). Estas a su vez se desglosan en varias variables adimensionales que dependen de h. Comenzamos 
        con las variables mas sencillas de definir con respecto a h.
        
    
        Parameters
        ----------
        h : float
            Altura adimensional. Varia en el intervalo (tol, 1-tol)
        m : float
            Exponente del factor de fricción para el gas. Para flujos turbulentos m = 0.2 y para laminares m = 1
        n : float
            Exponente del factor de fricción para el liquido. Para flujos turbulentos n = 0.2 y para laminares n = 1
    
        Returns
        -------
        array de F1(h) y F2(h)
    
        """
        
        #Variable Auxiliar
        _2h1 = 2.*h-1.
    
        # Perimetro gas - pared
        S_g = arcos(_2h1)
        
        # Perimetro liquido - pared
        S_l = pi - S_g
        
        # Perimetro interfaz
        S_i = sqrt(1. - _2h1*_2h1)
        
        #Area liquido
        A_l = 0.25*(pi - S_g + _2h1*S_i)
        
        #Area gas
        A_g = 0.25*(S_g - _2h1*S_i)
        
        #Area total
        A = 0.25*pi
        
        #Velocidad del liquido
        u_l = A/A_l
        
        #Velocidad del gas
        u_g = A/A_g
        
        # Diametro  hidraulico del liquido
        D_l = 4.*A_l/S_l
        
        #Diametro hidraulico del gas
        D_g = 4.*A_g/(S_g + S_i)
        
        
        #Funciones F1(h) y F2(h)
        f1_h = power(u_l*D_l, -n)*u_l*u_l*S_l/A_l
        f2_h = power(u_g*D_g, -m)*u_g*u_g*(S_g/A_g + S_i/A_l + S_i/A_g)
        
        return np.asarray((f1_h, f2_h)).T
    
    
    #Array de los valores de la altura adimensional en el rango (0,1). Para evitar problemas en h = 0 y h = 1 se incluye una 
    #tolerancia.
    h = np.linspace(tol, 1. - tol, puntos)
    
    #Array del Factor peso vs gradiente de presión
    Y = np.array((-1e4, -1e3, -1e2, -1e1, 0., 1, 4, 5, 10, 1e2, 1e3))
    

    ''' ----------------------*
    ~  TURBULENTO/TURBULENTO  ~
    *-----------------------'''
    
    #Exponentes del factor de fricción
    m = 0.2
    n = 0.2
    
    #Funciones que dependen de h
    Fh_tt = Funciones_h(h, m, n)

    #Parámetro de Lockhart and Martinelli
    X_tt = np.zeros(puntos)
    
    
    ''' ----------------------*
    ~   TURBULENTO/LAMINAR    ~
    *-----------------------'''
    
    #Exponentes del factor de fricción
    m = 1.0
    n = 0.2
    
    #Funciones que dependen de h
    Fh_tl = Funciones_h(h, m, n)

    #Parámetro de Lockhart and Martinelli
    X_tl = np.zeros(puntos)
    
    
    ''' ----------------------*
    ~   GRAFICACION Y AJUSTE  ~
    *-----------------------'''
    
    plt.clf()
    plt.xlabel('X')
    plt.ylabel('h/D')
    plt.suptitle('Nivel de Equilibrio para Flujo Estratificado')
    plt.title(' _ _  Tur/Lam     ___ Tur/Tur')
    plt.xlim((1e-3,1e4))
    plt.ylim((0,1))
    plt.xscale("log")
    #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    #g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    #fmt = mticker.FuncFormatter(g)
    colors = cm.rainbow(np.linspace(0, 1, len(Y)))
    
    
    
    #Ciclo de barrido de Y
    for y_i in range(len(Y)):

        #Ciclo de barrido de h
        for i in range(puntos):
            
            #turbulento - turbulento
            tmp = ( Fh_tt[i][1] - 4.*Y[y_i] ) / Fh_tt[i][0]
            X_tt[i] = sqrt(max(tmp,0))

            #turbulento - laminar
            tmp = ( Fh_tl[i][1] - 4.*Y[y_i] ) / Fh_tl[i][0]
            X_tl[i] = sqrt(max(tmp,0))
            
        plt.plot(X_tt, h, '-', label="{}".format(Y[y_i]), c=colors[y_i])
        plt.plot(X_tl, h, '--', c=colors[y_i])

    plt.legend(loc='best', fancybox=True, title='Y =')
    plt.show()

    
if 'name'== __main__():
    __main__()