# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt

m_proton = 1.67262192e-27 # kg
m_SrF = (88 + 19)*m_proton

k_B = 1.380649e-23 # J/K
C_F0 = 286 * 1e-6 * 1e-4 * 1e-6 * k_B # s * m^2

LAMBDA = 1064e-9 # m
WAVENUMBER = 2*pi/LAMBDA

g = 9.8 # m/s^2

def z_R(waist):
    """
    Returns the Rayleigh range in m.

    Parameters
    ----------

    waist : float
    waist in m.

    Returns
    -------
    z_R : float
    Rayleigh range in m.
    """
    return pi*waist**2/LAMBDA

def U_t(waist, power):
    """
    Returns the trap depth in J.

    Parameters
    ----------
    waist : float
    Waist in m.

    power : float
    Power in W.

    Returns
    -------
    U_t : float
    U_t in J.
    """
    return 2*power*C_F0 / (pi*waist**2)

def omega_z(waist, power):
    """
    z trapping frequency in Hz.

    Parameters
    ----------
    waist : float
    Waist in m.

    power : float
    Power in MW.

    Returns
    -------
    omega_z : float
    Trapping frequency in Hz
    """
    return sqrt(2*U_t(waist,power)*WAVENUMBER**2/(m_SrF))

def omega_r(waist, power):
    """
    r trapping frequency in Hz.

    Parameters
    ----------
    waist : float
    Waist in m.

    power : float
    Power in MW.

    Returns
    -------
    omega_r : float
    """
    return sqrt(4*U_t(waist,power)/(m_SrF*waist**2))

def z_eq(waist, power):
    """
    Returns gravity-displaced equilibrium z-coordinate.

    Parameters
    ----------
    waist : float
    Waist in m.

    power : float
    Power in MW.

    Returns
    -------
    z_eq : float
    z_eq in m.
    """
    return g/omega_z(waist,power)**2

def r_eq(waist, power):
    """
    Returns gravity-displaced equilibrium r-coordinate.

    Parameters
    ----------
    waist : float
    Waist in m.

    power : float
    Power in MW.

    Returns
    -------
    r_eq : float
    """
    return g/omega_r(waist,power)**2

def plot_Zeq_LAMBDA(finesse, in_power_arr):
    """
    Plots z_eq/LAMBDA as a function of the waist, for several input powers.
    The cavity's finesse determines the circulating beam power.

    Parameters
    ----------
    finesse : float
    Cavity's finesse.

    in_power_arr : np.ndarray
    Input power in mW.

    Returns
    -------
    None.
    """
    waist_arr = np.linspace(10, 100, 500)*1e-6 # m
    power_arr = (finesse/pi) * in_power_arr * 1e-3 # W

    fig, ax = plt.subplots()
    
    for power, in_power in zip(power_arr, in_power_arr):
        y = z_eq(waist_arr, power)/LAMBDA
        ax.plot(waist_arr*1e6, y, label=r'${0:.0f}$ $mW$'.format(in_power))

    ax.set_xlabel(r'Waist $[\mu m]$')
    ax.set_ylabel(r'$z_{eq}/\lambda$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(r'Equilibrium position and input power $(\mathcal{F}=$'+str(finesse)+')')
    ax.legend(title='Input power')

    #ax.grid()

    plt.show()

def plot_req_Zr(finesse, in_power_arr):
    """
    Plots r_eq/waist as a function of the waist, for several input powers.
    The cavity's finesse determines the circulating beam power.

    Parameters
    ----------
    finesse : float
    Cavity's finesse.

    in_power_arr : np.ndarray
    Input power in mW.

    Returns
    -------
    None.
    """
    waist_arr = np.linspace(10, 100, 500)*1e-6 # m
    power_arr = (finesse/pi) * in_power_arr * 1e-3 # W

    fig, ax = plt.subplots()
    
    for power, in_power in zip(power_arr, in_power_arr):
        y = r_eq(waist_arr, power)/waist_arr
        ax.plot(waist_arr*1e6, y, label=r'${0:.0f}$ $mW$'.format(in_power))

    ax.set_xlabel(r'Waist $[\mu m]$')
    ax.set_ylabel(r'$r_{eq}/w_{0}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(r'Equilibrium position and input power $(\mathcal{F}=$'+str(finesse)+')')
    ax.legend(title='Input power')

    #ax.grid()

    plt.show()
    
def plot_Utrap(finesse, in_power_arr):
    """
    Plots U_t as a function of the waist, for several input powers.
    The cavity's finesse determines the circulating beam power.

    Parameters
    ----------
    finesse : float
    Cavity's finesse.

    in_power_arr : np.ndarray
    Input power in mW.

    Returns
    -------
    None.
    """
    waist_arr = np.linspace(10, 100, 500)*1e-6 # m
    power_arr = (finesse/pi) * in_power_arr * 1e-3 # W

    fig, ax = plt.subplots()
    
    for power, in_power in zip(power_arr, in_power_arr):
        y = U_t(waist_arr, power)/k_B * 1e6 # uK
        ax.plot(waist_arr*1e6, y, label=r'${0:.0f}$ $mW$'.format(in_power))

    ax.set_xlabel(r'Waist $[\mu m]$')
    ax.set_ylabel(r'$U_{trap}$ $[\mu K]$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(r'Trap depth and input power $(\mathcal{F}=$'+str(finesse)+')')
    ax.legend(title='Input power')

    #ax.grid()
    
    plt.show()

def plot_omega_i(finesse, w_arr, z=True):
    """
    Plots U_t as a function of the input power for several beam waists.
    The cavity's finesse determines the circulating beam power.

    Parameters
    ----------
    finesse : float
    Cavity's finesse.

    w_arr : np.ndarray
    Beam waist in um.

    Returns
    -------
    None.
    """
    waist_arr = w_arr*1e-6 # m
    in_power_arr = np.linspace(5, 50, 200) * 1e-3 # W
    power_arr = (finesse/pi) * in_power_arr

    fig, ax = plt.subplots()
    
    for waist in waist_arr:
        if z:
            y = omega_z(waist, power_arr) * 1e-3 # kHz
        else:
            y = omega_r(waist, power_arr) * 1e-3 # kHz
            
        ax.plot(in_power_arr*1e3, y, label=r'${0:.0f}$ $\mu m$'.format(waist*1e6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r' Input Power $[mW]$')
    if z:
        ax.set_ylabel(r'$\omega_{z}$ $[kHz]$')
        ax.set_title(r'Radial trapping frequency and waist $(\mathcal{F}=$'+str(finesse)+')')
    else:
        ax.set_ylabel(r'$\omega_{r}$ $[kHz]$')
        ax.set_title(r'Radial trapping frequency and waist $(\mathcal{F}=$'+str(finesse)+')')
    
    ax.legend(title='Waist size')

    #ax.grid()
    
    plt.show()

finesse = 5000
in_power_arr = np.array([1, 5, 10, 20]) # mW
w_arr = np.array([10, 20, 30, 40], dtype=float) # um

plot_Zeq_LAMBDA(finesse, in_power_arr)
plot_req_Zr(finesse, in_power_arr)
plot_Utrap(finesse, in_power_arr)
plot_omega_i(finesse, w_arr, z=True)
plot_omega_i(finesse, w_arr, z=False)