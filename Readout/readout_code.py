# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:31:36 2024

@author: Alejandro
"""
import numpy as np
from numpy import sqrt, pi
import matplotlib.pyplot as plt
from scipy.integrate import quad

c = 2.997e8 # micrometers * MHz
wavelength = 663.3e-3 # Coupling wavelength in micrometers

class Cavity(object):
    def __init__(self, L, T1, T2, R1, R2, eta, eta_max, omega_c, omega_a, kappa, gamma):
        """
        Cavity object to perform experiments. It is initialized with the cavity's physical
        parameters, and the properties of the atom's transition.
        
        Parameters
        ----------
        L : float
        Cavity's length in micrometers.
        
        T1 : float
        Transmissivity of the input mirror.
        
        T2 : float
        Transmissivity of the output mirror.
        
        R1 : float
        Reflectivity of the input mirror on the inner face. It is not 1-T1 due to losses.
        
        R2 : float
        Reflectivity of the output mirror on the inner face. It is not 1-T2 due to losses.
        
        eta : float
        Cavity's effective cooperativity.
        
        eta_max : float
        Cavity's theoretical maximum cooperativity.
        
        omega_c : float
        Resonance frequency of the empty cavity in MHz.
        
        omega_a : float
        Atom's resonance frequency in MHz.
        
        kappa : float
        Cavity's resonance linewidth in MHz.
        
        gamma : float
        Atom's transition linewidth in MHz.
        
        """
        self.L = L
        
        self.T1 = T1
        self.T2 = T2
        
        self.R1 = R1
        self.R2 = R2
        
        self.finesse = pi/(1 - sqrt(R1*R2)) # Maximum theoretical finesse
        
        self.eta = eta
        self.eta_max = eta_max
        
        self.omega_c = omega_c
        self.omega_a = omega_a
        
        self.kappa = kappa
        self.gamma = gamma
        
        self.wavenumber = omega_c/c
        self.wavelength = 2*pi/self.wavenumber
        
        self.waist = sqrt(24*self.finesse/(pi*self.eta_max*self.wavenumber**2))
        
    def xi(self, omega_l, omega_i, FWHM):
        """
        Laser's detuning from omega_i, normalized to the resonance's FWHM.
        
        Parameters
        ----------
        omega_l : float
        Laser's frequency.
        
        omega_i: float
        Resonance frequency.
        
        FWHM : float
        Full Width at Half Maximum
        
        Returns
        -------
        xi : float
        Laser's detuning from omega_i, normalized to the resonance's FWHM.
        """
        return (omega_l-omega_i)/(FWHM/2)
    
    def La(self, omega_l):
        """
        Absorptive Lorentzian Lineshape.
        
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
        
        Returns
        -------
        La : float
        Absorptive Lorentzian Lineshape.
        """
        xa = self.xi(omega_l, self.omega_a, self.gamma)
        
        return 1/(1+xa**2)
    
    def Ld(self, omega_l):
        """
        Dispersive Lorentzian Lineshape.
        
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
        
        Returns
        -------
        Ld : float
        Dispersive Lorentzian Lineshape.
        """
        xa = self.xi(omega_l, self.omega_a, self.gamma)
        
        return -xa/(1+xa**2)
    
    def Rabi_splitting(self, N):
        """
        Rabi splitting of a cavity-atoms resonant system.
        
        Parameters
        ----------
        N : float
        Number of atoms.
        
        Returns
        -------
        Rabi_splitting : float
        Resonance's Rabi splitting.
        """
        return sqrt(N*self.eta*self.gamma*self.kappa)
    
    def T0(self, omega_l, N):
        """
        Power transmission for a symmetric cavity.
    
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
    
        N : float
        Numbers of atoms.
    
        Returns
        -------
        T0 : float
        Power transmission at each laser frequency.
        """
        LA = self.La(omega_l)
        LD = self.Ld(omega_l)
        xc = self.xi(omega_l, self.omega_c, self.kappa)
        
        return 1/((1+N*self.eta*LA)**2 + (xc+N*self.eta*LD)**2)

    def Ec(self, omega_l, N):
        """
        Intracavity intensity to incident intensity.
    
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
    
        N : float
        Numbers of atoms.
    
        Returns
        -------
        Ec : float
        Ec/Ein at each laser frequency.
        """
        return self.T1*(self.finesse/pi)**2*self.T0(omega_l, N)
        
    def T(self, omega_l, N):
        """
        Power transmission for an asymmetric cavity.
    
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
    
        N : float
        Numbers of atoms.
    
        Returns
        -------
        Power transmission at each laser frequency.
        """
    
        return self.T1*self.T2*(self.finesse/pi)**2*self.T0(omega_l, N)
    
    def R(self, omega_l, N):
        """
        Reflected power.
    
        Parameters
        ----------
        omega_l : float
        Laser's frequency in MHz.
    
        N : float
        Numbers of atoms.
    
        Returns
        -------
        Reflection power at each laser frequency.
        """
        # Length_cavity = 90287*pi*c/omega_a_0
        k_L = (omega_l/c)*self.L
        
        LA = self.La(omega_l)
        LD = self.Ld(omega_l)
        xc = self.xi(omega_l, self.omega_c, self.kappa)
        
        r1, r2 = sqrt(self.R1), sqrt(self.R2)
        a = r1
        b = (self.finesse/pi)*(self.T1*r2*np.exp(2j*k_L))/((1+N*self.eta*LA) - 1j*(xc+N*self.eta*LD))
        
        return np.real((a-b)*np.conjugate(a-b))
    
    def S(self, omega_l, N):
        """
        Scattered power.
        
        Parameters
        ----------
        omega_l : float
        Laser's frequency.
    
        N : float
        Numbers of atoms.
    
        Returns
        -------
        Reflected power at each laser frequency.
        """
        LA = self.La(omega_l)
        
        TP = self.T(omega_l, N)
        
        return TP*N*self.eta*LA*(self.T1+self.T2)/self.T2

    def n_phot_scatt(self, omega_l, N, nphot, trans=True):
        """
        Number of scattered photons per transmitted or reflected photon.
        
        Parameters
        ----------
        omega_l : float
        Probe laser's angular frequency in MHz.
    
        N : float
        Numbers of atoms.
        
        nphot : float
        Number of transmitted or reflected photons.
    
        trans : bool
        If True, the transmission in measured; if False, the reflection.
        
        Returns
        -------
        Number of photons scattered by the atoms.
        """
        
        TP = self.T(omega_l, N)
        SP = self.S(omega_l, N) # Scattering per incident photon
        
        if trans:
            return (SP/TP)*nphot # Scattering per transmitted photon
        else:
            RP = self.R(omega_l, N) 
            return (SP/RP)*nphot # Scattering per reflected photon
            
    def F_meas(self, omega_l, N, trans=True):
        """
        Total Fisher Information (phase and amp) per photon for transmission or 
        reflection.
        
        Parameters
        ----------
        omega_l : float
        Probe's frequency in MHz.
    
        N : float
        Numbers of atoms.
    
        trans : bool
        If True, the transmission in measured; if False, the reflection.
        
        Returns
        -------
        Total Fisher Information per photon.
        """
    
        LA = self.La(omega_l)
        
        if trans:
            TP0 = self.T0(omega_l, N)
            return 4*self.eta**2*TP0*LA
        else:
            Ttot = self.T1*self.R2
            TP0 = self.T0(omega_l, N)
            RP = self.R(omega_l, N)
            return 4*(Ttot*self.T1*(self.finesse/pi)**2/RP)*self.eta**2*TP0**2*LA
        
    def scR(self, omega_l, N_dw, det_dw):
        """
        Total scattering from down atoms.
        
        Parameters
        ----------
        omega_l : float
        Probe laser's angular frequency.
        
        N_dw : float
        Number of down atoms.
        
        det_dw : float
        Frequency difference between the excited states |1/2> and |3/2>.
        det_dw = omega_(1/2) - omega_(3/2)
        
        Returns
        -------
        scR : float
        Total scattering from down atoms.
        """
        xa_dw = self.xi(omega_l, self.omega_a+det_dw, self.gamma)
        #return (self.kappa)*N_dw*self.eta/xa_dw**2
        return (self.kappa/2/pi)*N_dw*self.eta/xa_dw**2
    
    def scRvar(self, omega_l, N_dw, det_dw, absrp, decay):
        """
        Total undesired repumping ~ extra variance, inducing noise per scatt photon.
        
        Parameters
        ----------
        omega_l : float
        Probe laser's angular frequency.
        
        N_dw : float
        Number of down atoms.
        
        det_dw : float
        Frequency difference between the excited states |1/2> and |3/2>.
        det_dw = omega_(3/2) - omega_(1/2)
        
        absrp : float
        Probability of a down photon, instead of an up one absorbing a sigma_plus photon.
        
        decay : float
        Probability of a |3/2, 1/2> atom emitting a pi photon, instead of sigma_plus one.
        
        Returns
        -------
        ns : float
        Number of scattered photons.
        """
        SCR = self.scR(omega_l, N_dw, det_dw)
        
        return (absrp*decay)**2*SCR
        
    def DN(self, f_meas, n_detec, n_scatt, b):
        """
        Variance in the measurement of the number of atoms.
        It adds the photon shot noise contribution from the detected photon's Fisher 
        Information and the Raman Scattering noise.
        
        Parameters
        ----------
        f_meas : float
        Fisher Information per detected photon.
        
        n_detec : float
        Number of detected photons.
        
        n_scatt : float
        Number of scattered photons.
        
        b : float
        Probability of Raman flip due to scattered photon.
        
        Returns
        -------
        DN : float
        Variance in the measurement of the number of atoms.
        """
        return 1/(f_meas*n_detec) + n_scatt*b
    
    def integral_nt(self, N, lim):
        """
        Returns the integral of T0 from -\infty to \infty. It is necessary to compute
        the number of transmitted photons during a scan measurement.
        
        Parameters
        ----------
        N : float
        Number of atoms.
        
        lim : float
        Integration limit (-lim, lim).
        
        Returns
        -------
        Integral of T0
        """
        # Function to integrate
        y = lambda omega_l: self.T0(omega_l, N)
    
        return quad(y, self.omega_a-lim, self.omega_a+lim, limit=500)[0]
    
    def integral_chirp(self, N, lim):
        """
        Returns the interal of a function needed to compute the Fisher information
        per photon in a chirp scan measurement.
        
        Parameters
        ----------
        N : float
        Number of atoms.
        
        lim : float
        Integration limit (-lim, lim).
        
        Returns
        -------
        Integral of the function.
        """
        # Function to integrate
        y = lambda omega_l: self.La(omega_l)*self.T0(omega_l, N)**2
        
        return quad(y, self.omega_a-lim, self.omega_a+lim, limit=500)[0]
    
    def integral_scattered(self, N_dw, det_dw, absrp, decay, lim):
        """
        Integral to compute the number of scattered photons in a scan measurement.
        
        Parameters
        ----------
        N_dw : float
        Number of down atoms.
        
        det_dw : float
        Frequency difference between the excited states |1/2> and |3/2>.
        det_dw = omega_(3/2) - omega_(1/2)
        
        absrp : float
        Probability of a down photon, instead of an up one absorbing a sigma_plus photon.
        
        decay : float
        Probability of a |3/2, 1/2> atom emitting a pi photon, instead of sigma_plus one.
        
        lim : float
        Integration limit (-lim, lim).
        
        Returns
        -------
        Integral.
        """
        # Function to integrate
        y = lambda omega_l: self.T0(omega_l, N_dw)*self.scRvar(omega_l, N_dw, det_dw, absrp, decay)
        
        return quad(y, self.omega_a-lim, self.omega_a+lim, limit=500)[0]
    
    def scan_scR(self, N_dw, det_dw, absrp, decay, lim):
        """
        Number of scattered photons in a scan measurement.
        
        Parameters
        ----------
        N : float
        Number of atoms.
        
        n_trans : float
        Number of transmitted photons.
        
        Returns
        -------
        ns : float
        Number of scattered photons.
        """
        integral_scatt = self.integral_scattered(N_dw, det_dw, absrp, decay, lim)
        integral_trans = self.integral_nt(N_dw, lim)
        
        return integral_scatt/integral_trans
    
    def plot_T_T0(self, omega_l, N, plt_T=True, resonance=True, production_figure=False, logy=False):
        """
        Plots the cavity's transmission.
        
        Parameters
        ----------
        omega_l : np.ndarray
        Probe laser's angular frequency in MHz.
        
        N : float
        Number of atoms.
        
        plt_T : bool
        If True, plot the asymmetric cavity transmission; if True, the symmetric.
        If the input and output mirrors are identical, the two values are the same.
        
        resonance : bool
        If True, plots a dashed black line at the unshifted cavity resonance.
        
        production_figure : bool
        If True, show publication-quality plots and save them as a pdf.
        
        Returns
        -------
        None.
        """
        if plt_T:
            T_arr = self.T(omega_l, N)
        else:
            T_arr = self.T0(omega_l, N)
            
        Delta = (omega_l - self.omega_a)/2/pi
        
        if production_figure:
            plt.rcParams.update({'font.size':14})
            fig, ax = plt.subplots(figsize=(6.34,3.94), dpi=500)
        else:
            fig, ax = plt.subplots()
        
        ax.plot(Delta, T_arr, 'k-')
        if resonance:
            ax.plot(((self.omega_c-self.omega_a)/2/pi, (self.omega_c-self.omega_a)/2/pi), (ax.get_ylim()[0], ax.get_ylim()[-1]), 'k--')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('$\\Delta/2\\pi$ $[MHz]$')
        if plt_T:
            ax.set_ylabel('$\\mathcal{T}$')
        else:
            ax.set_ylabel('$\\mathcal{T}_{0}$')
        ax.set_title('Transmission $(N={0:.0f})$'.format(N))
        
        ax.margins(0)
        
        plt.tight_layout()
        
        plt.show()
    
    def plot_R(self, omega_l, N, resonance=True, production_figure=False, logy=False):
        """
        Plots the cavity's reflection spectrum.
        
        Parameters
        ----------
        omega_l : np.ndarray
        Probe laser's angular frequency in MHz.
        
        N : float
        Number of atoms.
        
        resonance : bool
        If True, plots a dashed black line at the unshifted cavity resonance.
        
        production_figure : bool
        If True, show publication-quality plots and save them as a pdf.
        
        Returns
        -------
        None.
        """
        R_arr = self.R(omega_l, N)
            
        Delta = (omega_l - self.omega_a)/2/pi
        
        if production_figure:
            plt.rcParams.update({'font.size':14})
            fig, ax = plt.subplots(figsize=(6.34,3.94), dpi=500)
        else:
            fig, ax = plt.subplots()
        
        ax.plot(Delta, R_arr, 'k-')
        if resonance:
            ax.plot(((self.omega_c-self.omega_a)/2/pi, (self.omega_c-self.omega_a)/2/pi), (ax.get_ylim()[0], ax.get_ylim()[-1]), 'k--')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('$\\Delta/2\\pi$ $[MHz]$')
        ax.set_ylabel('$\\mathcal{R}$')
        ax.set_title('Reflectance $(N={0:.0f})$'.format(N))
        
        ax.margins(0)
        
        plt.tight_layout()
        
        plt.show()

    def plot_ns(self, omega_l, N, trans=True, Rm=3.5, resonance=True, production_figure=False, logy=False):
        """
        Plots the number of Raman/Rayleigh scattered photon per reflected or transmitted photon.
        
        Parameters
        ----------
        omega_l : np.ndarray
        Probe laser's angular frequency in MHz.
        
        N : float
        Number of atoms.

        trans : bool
        If True, transmission; else, reflection.

        Rm : float
        Ratio of Raman per Rayleigh.
        
        resonance : bool
        If True, plots a dashed black line at the unshifted cavity resonance.
        
        production_figure : bool
        If True, show publication-quality plots and save them as a pdf.
        
        logy : bool
        If True, use log scale for y.
        
        Returns
        -------
        None.
        """
        ns_arr = Rm*self.n_phot_scatt(omega_l, N, 1, trans=trans)
            
        Delta = (omega_l - self.omega_a)/2/pi
        
        if production_figure:
            plt.rcParams.update({'font.size':14})
            fig, ax = plt.subplots(figsize=(6.34,3.94), dpi=500)
        else:
            fig, ax = plt.subplots()
        
        ax.plot(Delta, ns_arr, 'k-')
        if resonance:
            ax.plot(((self.omega_c-self.omega_a)/2/pi, (self.omega_c-self.omega_a)/2/pi), (ax.get_ylim()[0], ax.get_ylim()[-1]), 'k--')
        
        if logy:
            ax.set_yscale('log')
        
        ax.set_xlabel('$\\Delta/2\\pi$ $[MHz]$')
        
        if trans:
            ax.set_ylabel('$\\mathcal{S}/\\mathcal{T}$')
        else:
            ax.set_ylabel('$\\mathcal{S}/\\mathcal{R}$')

        if Rm==1:
            ax.set_title('Rayleigh Scattering $(N={0:.0f})$'.format(N))
        else:
            ax.set_title('Raman Scattering $(N={0:.0f})$'.format(N))
            
        ax.margins(0)
        
        plt.tight_layout()
        
        plt.show()
        
    def plot_Ec(self, omega_l, N, resonance=True, production_figure=False, logy=False):
        """
        Plots the intracavity intensity.
        
        Parameters
        ----------
        omega_l : np.ndarray
        Probe laser's angular frequency in MHz.
        
        N : float
        Number of atoms.
        
        resonance : bool
        If True, plots a dashed black line at the unshifted cavity resonance.
        
        production_figure : bool
        If True, show publication-quality plots and save them as a pdf.
        
        logy : bool
        If True, use log scale for y.
        
        Returns
        -------
        None.
        """
        Ec_arr = self.Ec(omega_l, N)
            
        Delta = (omega_l - self.omega_a)/2/pi
        
        if production_figure:
            plt.rcParams.update({'font.size':14})
            fig, ax = plt.subplots(figsize=(6.34,3.94), dpi=500)
        else:
            fig, ax = plt.subplots()
        
        ax.plot(Delta, Ec_arr, 'k-')
        if resonance:
            ax.plot(((self.omega_c-self.omega_a)/2/pi, (self.omega_c-self.omega_a)/2/pi), (ax.get_ylim()[0], ax.get_ylim()[-1]), 'k--')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\Delta/2\pi$ $[MHz]$')
        ax.set_ylabel(r'$\mathcal{E}_{c}$')
        ax.set_title('Intracavity field $(N={0:.0f})$'.format(N))
        
        ax.margins(0)
        
        plt.tight_layout()
        
        plt.show()
        
    def FI_per_phot(self, omega_l, N, trans=True, production_figure=False, logy=False):
        """
        Plots the total Fisher information per transmitted/reflected photon.
        
        Parameters
        ----------
        omega_l : np.ndarray
        Probe frequency values where the Fisher information is computed.
        If False, the function automatically initializes a convinient one.
        
        N : float
        Number of atoms.
        
        trans : bool
        If True, plot FI per transmitted photon; otherwise, FI per reflected photon.
        
        production_figure : bool
        If True, show publication-quality plots and save them as a pdf.
        
        logy : bool
        If True, use log scale for y.
        
        Returns
        -------
        None
        """
        
        f_meas  = self.F_meas(omega_l, N, trans)
    
        x = (omega_l-self.omega_a)/2/pi #/(self.kappa/2)
        
        if production_figure:
            plt.rcParams.update({'font.size':14})
            fig_FI, ax_FI = plt.subplots(figsize=(6.34,3.94), dpi=500)
        else:
            fig_FI, ax_FI = plt.subplots()
            
        ax_FI.plot(x, f_meas*1e5, 'k-', label='$F_{tot}$')
        if logy:
            ax_FI.set_yscale('log')
        ax_FI.legend()
        ax_FI.set_ylabel('$QFI$  $[\\times 10^{-5}]$')
        ax_FI.set_xlabel('$\\Delta / 2\\pi [MHz]$')
        #ax_FI.set_xlabel('$2\\Delta/\\kappa$')
        if trans:
            ax_FI.set_title('Fisher Information per transmitted photon')
        else:
            ax_FI.set_title('Fisher Information per reflected photon')
            
        plt.tight_layout()
    
        plt.show()
        
    def scan_measurement(self, N, det_dw, absrp, decay, limit, q_efficiency=1):
        """
        Atom number variance as a function of the number of detected photons.
        Assume an amplitude measurement where the probe's frequency is 
        swept through the peaks.

        Parameters
        ----------
        self : Cavity
        A Cavity object where the experiment is performed.

        N : float
        Number of atoms.
        
        det_dw : float
        Splitting of hyperfine ground state.

        b : float
        Raman scattering rate into other levels.

        q_efficiency : float
        Detector's efficiency.

        Returns
        -------
        Atom number variance as a function of the number of detected photons.
        """
        integral_trans = self.integral_nt(N, limit)
        integral_FI    = self.integral_chirp(N, limit)

        f_meas = 4*self.eta**2*integral_FI/integral_trans
        print(f_meas, integral_trans, integral_FI)
        if N < 400:
            n_detec = np.linspace(0.1, 500, 1000)
        else:
            n_detec = np.linspace(4, 4000, 1000)

        n_trans = n_detec/q_efficiency

        raman_var = self.scan_scR(N, det_dw, absrp, decay, limit)

        uncert = self.DN(f_meas, n_detec, n_trans, raman_var)

        fig_DN, ax_DN = plt.subplots(ncols=1)
        ax_DN.plot(n_detec, uncert, 'k-')
        ax_DN.plot((n_detec[0], n_detec[-1]), (1, 1), 'k--')
        ax_DN.set_xscale('log')
        ax_DN.set_yscale('log')
        ax_DN.set_ylabel('$(\\Delta N)^{2}$')
        ax_DN.set_xlabel('Number of detected photons')
        ax_DN.set_title('Atom Number Variance [Amplitude Measurement, $N_{\\uparrow)$'+'$={0:d}$]'.format(int(N)))

        ax_DN.legend()
        ax_DN.grid()
        ax_DN.margins(0) 
        ax_DN.tick_params(axis='both', which='both', length=3, width=1, labelsize=10)

        ax_sigma = ax_DN.twinx()
        ax_sigma.set_ylim(ax_DN.get_ylim()[0]/(2*N/4), ax_DN.get_ylim()[1]/(2*N/4))
        ax_sigma.figure.canvas.draw()
        ax_sigma.set_ylabel('$\\frac{(\\Delta N)^{2}}{(\\Delta N)^{2}_{SQL}}$')
        ax_sigma.set_yscale('log')
        ax_sigma.tick_params(axis='both', which='both', length=3, width=1, labelsize=10)

        plt.tight_layout()

        plt.show()

def waist_from_L_R(L, R, omega_c):
    """
    Computes the waist from the caivty's length and the mirrors' ROC.

    Parameters
    ----------
    L : float
        Cavity's length in cm.
    R : float
        ROC in cm.
    omega_c : float
        Angular resonance frequency in MHz.
        
    Returns
    -------
    Waist in micrometers.

    """
    k = omega_c/c # Wavenumber
    return sqrt(sqrt(L*(2*R-L))/k)

def waist_from_coop_F(eta, finesse, omega_c):
    """
    Returns the waist from the cavity's finesse and ideal, maximum cooperativity.

    Parameters
    ----------
    eta : float
        Cavity's cooperativity.
    finesse : float
        Cavity's finesse.
    omega_c : float
        Angular resonance frequency in MHz.

    Returns
    -------
    Waist in micrometers.

    """
    k = omega_c/c # Wavenumber
    return sqrt(24*finesse/pi/eta/k**2)
 
def cooperativity(finesse, waist, omega_c):
    """
    Theoretical maximum cooperativity from the finesse, waist and angular frequency.

    Parameters
    ----------
    finesse : float
        Cavity's finesse.
    waist : float
        Cavity's waist in um.
    omega_c : float
        Angular resonance frequency in MHz.

    Returns
    -------
    THeoretical maximum cooperativity.

    """
    k = omega_c/c # Wavenumber
    return 24*finesse/(pi*waist**2*k**2)
    
def length_from_waist(R, waist, omega_c):
    """
    Returns the cavity's length from its waist and the mirrors' ROC.

    Parameters
    ----------
    R : float
        ROC in cm.
    waist : float
        Waist in micrometers.
    omega_c : float
        Angular resonance frequency in MHz.

    Returns
    -------
    Length in MICROMETERS.

    """
    k = omega_c/c # Wavenumber in um**-1
    R *= 1e4 # to um
    L = R + sqrt(R**2 - k**2*waist**4) 
    return L

def design(R, finesse, waist, omega_c, trans, loss=3.8e-6):
    """
    Returns the cavity's length and linewidth, and the mirrors' reflectance and 
    transmittance. The values are found to match the input parameters.
    
    Parameters
    ----------
    R : float
    ROC in cm.
    
    finesse : float
    Finesse.
    
    waist : float
        Waist in micrometers.
    
    omega_c : float
    Cavity's resonance frequency in MHz.
    
    trans : bool
    If True, assume a cavity for transmission experiments.
    
    loss : float, optional
    Mirrors' loss.
    
    Returns
    -------
    L : float
        Cavity's length
    T1 : float
        Transmitance of the input mirror.
    T2 : float
        Transmitance of the output mirror.
    R1 : float
        Reflectance of the input mirror.
    R2 : float
        Reflectance of the input mirror.
    omega_c_new : float
        Resonant frequency of the cavity. Note that arbitrary values of omega_c
        might be unphysical, since the cavity's length fixes the allowed resonant
        frequencies.
    
    """

    if trans:
        R1 = R2 = 1 - pi/finesse
        T1 = T2 = 1 - R2 - loss
    else:
        # Second mirror is a near-perfect reflector
        T2 = loss
        R2 = 1 - T2 - loss
        # The finesse determines the other reflectance
        R1 = (1 - pi/finesse)**2/R2
        T1 = 1 - R1 - loss

    # The length can be found from the waist       
    L = length_from_waist(R, waist, omega_c)

    # The cavity's freq must be an integer of pi*c/L
    n = omega_c*L/(pi*c)
    omega_c_new = np.round(n)*pi*c/L

    return L, T1, T2, R1, R2, omega_c_new

def meas(cav, n_detec, N, b, q, trans, power=False, raman_params=False, omega_laser=False, production_figure=False, ret_DN=False):
    """
    Atom number measurement uncertainty.

    Parameters
    ----------
    cav : Cavity.Cavity
    Cavity object where the experiments are performed.

    n_detec : np.ndarray
    Number of detected photons.

    N : float
    Total number of atoms in the cavity.

    b : float
    Probability of lossing an atom per scattered photon.

    q : float
    Detector efficiency.

    trans : bool
    If True, assumes transmission; otherwise, reflection.

    power : float
    Input power in watts.

    raman_params : dir
    Contains the detuning (det) between the ground states such that \Delta_2 = \Delta_1 + det,
    the ratio between the matrix elements (r), and the decay branching ratio (B) to the other state.

    omega_laser : float
    If given, the laser probe will be parked at this frequency. If not, the function
    will find the maximum Fisher Information position to place the probe.
    
    production_figure : bool
    If True, show publication-quality plots and save them as a pdf.
    
    ret_DN : float
    If True, return the uncertainty values.
    
    Returns
    -------
    DN : float
    Uncertainty.
    """
    #FI_arr = 2*cav.F_meas(omega_l_arr, N, trans=trans) # A chirp measurement measures the two peak

    N_up = N/2

    if not omega_laser:
        max_det = 0.6*sqrt(N_up*cav.eta*cav.gamma*cav.kappa) # Search peak FI between omega_c and a bit after the max splitting

        omega_l_arr = np.linspace(cav.omega_c, cav.omega_c+max_det, 2049)
        FI_arr = cav.F_meas(omega_l_arr, N_up, trans=trans)

        optm_pos = np.argmax(FI_arr) # Maximum Fisher Information

        f_meas = FI_arr[optm_pos]
        omega_l = omega_l_arr[optm_pos]
        print('Delta/2pi: ', (omega_l-cav.omega_a)/2/pi)

    else:
        omega_l = omega_laser
        f_meas = cav.F_meas(omega_l, N_up, trans=trans)
        print('Delta/2pi: ', (omega_l-cav.omega_a)/2/pi)

    n_out = n_detec/q

    n_scatt = cav.n_phot_scatt(omega_l, N_up, n_out, trans=trans) # Number of scattered photons

    DN = cav.DN(f_meas, n_detec, n_scatt, b)

    if raman_params:
        delta_up = omega_l-cav.omega_a
        delta_dw = delta_up + raman_params['det']
        prob = raman_params['B']*raman_params['r']**2*(delta_up/delta_dw)**2
        # Zilong's paper computes DJ/DJ_SQL, DN/DN_SQL should be the same expression as the factors of 2 cancel
        DN += (4*prob*n_scatt*3.5)*N/4 # Raman = 3.5*Rayleigh. Factor of N/4 to compute DN and not DN/DN_SQL
        
    if production_figure:
        plt.rcParams.update({'font.size':14})
        fig, ax = plt.subplots(figsize=(6.34,3.94), dpi=500)
    else:
        fig, ax = plt.subplots()

    ax.plot(n_out, DN, 'k-', label='$N={0:d}$'.format(int(N)))
    if trans:
        ax.set_xlabel('Number of transmitted photons')
    else:
        ax.set_xlabel('Number of reflected photons')
    ax.set_ylabel('$(\\Delta N)^{2}$')
    #ax.set_title('Uncertainty [$N={0:d}$]'.format(int(N)))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.margins(0)

    ax_twin = ax.twinx()
    ax_twin.set_ylim((ax.get_ylim()[0]/(N/4), ax.get_ylim()[1]/(N/4)))
    ax_twin.set_yscale('log')
    ax_twin.set_ylabel('$(\\Delta N)^{2}/SQL$')

    if power:
      # Factor to convert output photons to time in ms
      nu = 1e6*omega_l/2/pi # Frequency in Hz
      #print("Freq: ", nu)
      #print("Lambda: ", 1e9*3e8/nu)
      phot_to_ms = 1e3 * 1.054e-34 * nu / power # hbar in J*s
      if trans:
        phot_to_ms /= cav.T(omega_l, N_up)
      else:
        phot_to_ms /= cav.R(omega_l, N_up)
      ax_twin_y = ax.twiny()
      ax_twin_y.set_xlim((ax.get_xlim()[0]*phot_to_ms, ax.get_xlim()[1]*phot_to_ms))
      ax_twin_y.set_xscale('log')
      ax_twin_y.set_xlabel('Measurement time $[ms]$')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    if ret_DN:
        return DN, n_out

def optimal_point(cav, N, b, q, trans=True):
    """
    Find the point and the number of output photons at the optimal measurement point.

    Parameters
    ----------
    cav : Cavity
        Cavity object to do the experiment.
    N : float
        Number of molecules.
    b : float
        Raman scattering per Rayleigh scattering.
    q : float
        Detector efficiency.
        
    Returns
    -------
    n_out : float
        Number of output photons at the optimal point.
    DN : float
        Corresponfing uncertainty.
    trans : bool
    If True, transmission.
    
    """
    N_up = N/2
    
    max_det = 0.6*sqrt(N_up*cav.eta*cav.gamma*cav.kappa) # Search peak FI between omega_c and a bit after the max splitting

    omega_l_arr = np.linspace(cav.omega_c, cav.omega_c+max_det, 2049)
    FI_arr = cav.F_meas(omega_l_arr, N_up, trans=trans)

    optm_pos = np.argmax(FI_arr) # Maximum Fisher Information

    f_meas = FI_arr[optm_pos]
    omega_l = omega_l_arr[optm_pos]
    
    n_scatt = cav.n_phot_scatt(omega_l, N_up, 1, trans=trans) # Number of scattered photons per out photon
    
    return 1/sqrt(f_meas*b*n_scatt/q), 2*sqrt(b*n_scatt/q/f_meas)
    
def DN_finesse(finesse_list, waist_arr, omega_c, omega_a, kappa, Gamma):
    """
    Plots DN vs waist for several finesse values.

    Parameters
    ----------
    finesse_list : tuple
    Finesse values that will be used.
    
    waist_arr : np.ndarray
    Array of waist [um].
    
    The rest are self-explanatory.
    
    Returns
    -------
    None.

    """
    DN_list = []
    
    N = 1000
    b = 3.5
    q = 0.2
    
    for finesse in finesse_list:
        DN_arr = np.zeros_like(waist_arr)
        for i in range(len(waist_arr)):
            T = pi/finesse
            R = 1 - T
            
            eta_max = cooperativity(finesse, waist_arr[i], omega_c)
            eta_eff = eta_max/6
            
            L = pi*c/kappa/finesse
            
            omega_c = (pi*c/L)*np.round(omega_c*L/pi/c)
            
            cav = Cavity(L, T, T, R, R, eta_eff, eta_max, omega_c, omega_a, 
                         kappa, Gamma)
            DN_arr[i] = optimal_point(cav, N, b, q)[1]
        
        DN_list.append(DN_arr)
        
    fig, ax = plt.subplots()
    
    for i in range(len(finesse_list)):
        ax.plot(waist_arr, DN_list[i]/(N/4), label='{0:.0f}'.format(finesse_list[i]*1e-3))
    
    ax.set_xlabel(r'Waist $[\mu m]$')
    ax.set_ylabel(r'$(\Delta N)^2/SQL$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(title=r'Finesse $(\times 10^{3})$')
    
    plt.show()
    
def DN_waist(waist_list, finesse_arr, omega_c, omega_a, kappa, Gamma):
    """
    Plots DN vs waist for several finesse values.

    Parameters
    ----------
    finesse_list : tuple
    Finesse values that will be used.
    
    waist_arr : np.ndarray
    Array of waist [um].
    
    The rest are self-explanatory.
    
    Returns
    -------
    None.

    """
    DN_list = []
    
    N = 1000
    b = 3.5
    q = 0.2
    
    for waist in waist_list:
        DN_arr = np.zeros_like(finesse_arr)
        for i in range(len(finesse_arr)):
            finesse = finesse_arr[i]
            
            T = pi/finesse
            R = 1 - T
            
            L = pi*c/kappa/finesse
            
            omega_c = (pi*c/L)*np.round(omega_c*L/pi/c)
            
            eta_max = cooperativity(finesse, waist, omega_c)
            eta_eff = eta_max/6
            
            cav = Cavity(L, T, T, R, R, eta_eff, eta_max, omega_c, omega_a, 
                         kappa, Gamma)
            DN_arr[i] = optimal_point(cav, N, b, q)[1]
            print(i, finesse)
            print(i, "=", optimal_point(cav, N, b, q)[1]/(N/4))
        DN_list.append(DN_arr)
        
    fig, ax = plt.subplots()
    
    for i in range(len(waist_list)):
        ax.plot(finesse_arr*1e-3, DN_list[i]/(N/4), label=r'{0:.0f} $\mu m$'.format(waist_list[i]))
    
    ax.set_xlabel(r'Finesse $(\times 10^{3})$')
    ax.set_ylabel(r'$(\Delta N)^2/SQL$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(title=r'Waist $[\mu m]$')
    
    plt.show()

def test(finesse, waist, omega_a, omega_c, kappa, Gamma, N, b, q):
    T = pi/finesse
    R = 1 - T
    
    L = pi*c/kappa/finesse
    
    omega_c = (pi*c/L)*np.round(omega_c*L/pi/c)
    
    eta_max = cooperativity(finesse, waist, omega_c)
    eta_eff = eta_max/6
    
    test = Cavity(L, T, T, R, R, eta_eff, eta_max, omega_c, omega_a, 
                 kappa, Gamma)
    print(optimal_point(test, N, b, q)[1]/(N/4))

# Molecular properties (assume that of SrF)
omega_a = 2*pi*451.97e6 # MHz
Gamma = 2*pi* 6.6 # MHz
omega_c = omega_a + Gamma*500 # Detuned cavity

###
# Parameters for the short, 3 cm long cavity
finesse = 150e3
#ROC = 1.5 # Mirrors' radii of curvature in cm
waist = 10 # Waist in um of the near-concentric cavity


###########
# Parameters for the long, 5 cm long cavity
# finesse = 150e3
ROC = 2.5 # Mirrors' radii of curvature in cm
# waist = 10 # Waist in um of the near-concentric cavity
###########


L, T1, T2, R1, R2, omega_c_new = design(ROC, finesse, waist, omega_c, True)
kappa = pi*c/L/finesse

eta_max = cooperativity(finesse, waist, omega_c_new)
eta_eff = eta_max * (2/9) * (3/4) # The last two factors account for the BR and inhomogeneity reductions

wavenumber = omega_c/c
wavelength = 2*pi/wavenumber

# Initialize cavity object
SrF = Cavity(L, T1, T2, R1, R2, eta_eff, eta_max, omega_c_new, omega_a, kappa, Gamma)

L_cm = SrF.L*1e-4 # cm

# Double-check
print('The cavity\'s parameters are:',
     'Length = {0:.5f} cm = {1:.0f} um'.format(L_cm, L),
     'ΔL = {0:.0f} um'.format(2*ROC*1e4 - L),
     '(T1,R1) = ({0:.7f},{1:.7f})'.format(SrF.T1, SrF.R1),
     '(T2,R2) = ({0:.7f},{1:.7f})'.format(SrF.T2, SrF.R2),
     'finesse = {0:.1f}'.format(SrF.finesse), 
     'cooperativity = {0:.1f}'.format(SrF.eta),
     'detuning = {0:.1f} MHz = {1:.0f}\\Gamma'.format((SrF.omega_c-SrF.omega_a)/2/pi, (SrF.omega_c-SrF.omega_a)/SrF.gamma),
     'kappa = 2*pi*{0:.3f} kHz'.format(1e3*SrF.kappa/2/pi),
     'waist = {0:.3f} micrometers'.format(SrF.waist),
     sep='\n')

N = 1000 # Number of molecules
N_up = N/2
b = 3.5 # Assume 3.5 molecules are lost for each Rayleigh photon
q = 0.2 # Detectors quantum efficiency
power = 0.5e-14 # Input power in W

n_detec_1 = np.linspace(0.1, 800, 5000) # number of detected photons

# Parameters to estimate the Raman flip noise
# USE THESE FOR |N=0, F=1, m=0>
raman_params_0 = {'det':2*pi*110, 'B':2/9, 'r':sqrt(2)}
# USE THESE FOR |N=2, F=1, m=0>
raman_params_2 = {'det':2*pi*45e3, 'B':2/9, 'r':1/sqrt(2)}

lim = 4 # Interval to plot
omega_l = np.linspace(SrF.omega_c-lim, SrF.omega_c+lim, 500)

# Fisher information per transmitted photon
SrF.FI_per_phot(omega_l, N_up, trans=True, production_figure=0)
# Intracavity field
SrF.plot_Ec(omega_l, N_up, resonance=True, production_figure=0)
# Transmission spectrum
SrF.plot_T_T0(omega_l, N_up, plt_T=True, resonance=True, production_figure=0)
# Reflection spectrum
SrF.plot_R(omega_l, N_up, resonance=True, production_figure=0)
# Raman scattered photons per transmitted photon
SrF.plot_ns(omega_l, N_up, trans=True, Rm=3.5, resonance=True, production_figure=0, logy=0)
# Measurement
meas(SrF, n_detec_1, N, b, q, power=power, raman_params=False, trans=True, production_figure=0)
meas(SrF, n_detec_1, N, b, q, power=power, raman_params=raman_params_0, trans=True, production_figure=0)
meas(SrF, n_detec_1, N, b, q, power=power, raman_params=raman_params_2, trans=True, production_figure=0)
