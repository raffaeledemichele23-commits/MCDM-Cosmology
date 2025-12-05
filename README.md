# MCDM-Cosmology
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i0, k0, i1, k1

# ==========================================
# MCDM VALIDATION: NGC 6503 ROTATION CURVE
# Author: Raffaele De Michele (2025)
# ==========================================

# 1. PHYSICAL CONSTANTS
# ---------------------
# G converted to astronomical units: kpc * (km/s)^2 / M_sun
G = 4.302e-6  

# 2. OBSERVATIONAL DATA (NGC 6503)
# --------------------------------
# Source: Begeman (1991) via SPARC Database
r_data = np.array([1.26, 2.52, 3.78, 5.04, 6.30, 7.56, 8.82, 10.08,
                   11.34, 12.60, 13.86, 15.12, 16.38, 17.64, 18.90,
                   20.16, 21.42]) # Radius in kpc

v_obs = np.array([68.0, 95.0, 107.0, 113.0, 116.0, 117.0, 118.0, 118.0,
                  119.0, 120.0, 121.0, 121.0, 122.0, 122.0, 122.0,
                  122.0, 122.0]) # Velocity in km/s

v_err = np.full_like(v_obs, 3.0) # Estimated observational error

# 3. THEORETICAL MODELS
# ---------------------

def velocity_baryonic(r, M_star=3.8e9, M_HI=7.2e8, R_d=1.67):
    """
    Calculates the contribution of visible matter (Stars + Gas).
    - Disk: Freeman's exponential disk model
    - Gas: Simplified exponential distribution
    """
    # Numerical safety: avoid division by zero at r=0
    r_safe = np.where(r == 0, 1e-10, r)
    
    # Stellar Disk Contribution
    y = r_safe / (2 * R_d)
    B_term = i0(y) * k0(y) - i1(y) * k1(y)
    v_disk_sq = 4 * np.pi * G * (M_star / (2 * np.pi * R_d**2)) * R_d * y**2 * B_term
    
    # Gas Contribution
    v_gas_sq = G * M_HI * (1 - np.exp(-r_safe/R_d) * (1 + r_safe/R_d)) / r_safe
    
    return np.sqrt(np.abs(v_disk_sq + v_gas_sq))

def velocity_fluid_MCDM(r, R_gal, V_fluid_max):
    """
    MCDM FLUID VORTEX MODEL
    Represents the velocity field of a viscous fluid vortex.
    - Profile: Pseudo-isothermal (generates flat rotation curves)
    - R_gal: Vortex core radius (kpc)
    - V_fluid_max: Asymptotic velocity of the fluid flow (km/s)
    """
    r_safe = np.where(r == 0, 1e-10, r)
    
    # Formula deriving from Navier-Stokes for a vortex core
    term = 1 - (R_gal / r_safe) * np.arctan(r_safe / R_gal)
    v_sq = V_fluid_max**2 * term
    
    return np.sqrt(np.abs(v_sq))

def velocity_total_MCDM(r, R_gal, V_fluid_max):
    """Total velocity = sqrt( V_baryons^2 + V_fluid^2 )"""
    v_bar = velocity_baryonic(r)
    v_fluid = velocity_fluid_MCDM(r, R_gal, V_fluid_max)
    return np.sqrt(v_bar**2 + v_fluid**2)

# 4. FITTING PROCEDURE
# --------------------
print("Running non-linear least squares fit...")

# Initial guesses: R_gal=2 kpc, V_fluid=100 km/s
p0_guess = [2.0, 100.0]

try:
    popt, pcov = curve_fit(velocity_total_MCDM, r_data, v_obs, 
                           p0=p0_guess, sigma=v_err, absolute_sigma=True)
    
    R_best, V_best = popt
    perr = np.sqrt(np.diag(pcov)) # Parameter errors (1-sigma)
    
    # 5. CHI-SQUARE CALCULATION
    # -------------------------
    v_model = velocity_total_MCDM(r_data, *popt)
    chi2 = np.sum(((v_obs - v_model) / v_err)**2)
    dof = len(r_data) - 2 # Degrees of freedom (N points - 2 parameters)
    chi2_red = chi2 / dof

    # 6. RESULTS OUTPUT
    # -----------------
    print("\n" + "="*40)
    print("MCDM FIT RESULTS (NGC 6503)")
    print("="*40)
    print(f"R_gal (Vortex Core)     : {R_best:.2f} +/- {perr[0]:.2f} kpc")
    print(f"V_fluid (Asymptotic)    : {V_best:.2f} +/- {perr[1]:.2f} km/s")
    print("-" * 40)
    print(f"Reduced Chi-Square      : {chi2_red:.3f}")
    print("="*40)

    # 7. PLOTTING
    # -----------
    plt.figure(figsize=(10, 6))
    
    # Plot Data
    plt.errorbar(r_data, v_obs, yerr=v_err, fmt='ko', label='Observed Data (Begeman 1991)')
    
    # Plot Models
    r_plot = np.linspace(0.1, 22, 100) # Smooth line for plotting
    
    plt
