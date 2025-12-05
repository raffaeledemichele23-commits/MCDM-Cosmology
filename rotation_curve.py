import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i0, k0, i1, k1

# ==========================================
# MCDM VALIDATION: NGC 6503 ROTATION CURVE
# Author: Raffaele De Michele (2025)
# ==========================================

# 1. PHYSICAL CONSTANTS
G = 4.302e-6  # kpc * (km/s)^2 / M_sun

# 2. OBSERVATIONAL DATA (NGC 6503 - Begeman 1991)
r_data = np.array([1.26, 2.52, 3.78, 5.04, 6.30, 7.56, 8.82, 10.08,
                   11.34, 12.60, 13.86, 15.12, 16.38, 17.64, 18.90,
                   20.16, 21.42])
v_obs = np.array([68.0, 95.0, 107.0, 113.0, 116.0, 117.0, 118.0, 118.0,
                  119.0, 120.0, 121.0, 121.0, 122.0, 122.0, 122.0,
                  122.0, 122.0])
v_err = np.full_like(v_obs, 3.0)

# 3. MODELS
def velocity_baryonic(r, M_star=3.8e9, M_HI=7.2e8, R_d=1.67):
    # Numerical safety for r=0
    r_safe = np.where(r == 0, 1e-10, r)
    y = r_safe / (2 * R_d)
    B_term = i0(y) * k0(y) - i1(y) * k1(y)
    v_disk_sq = 4 * np.pi * G * (M_star / (2 * np.pi * R_d**2)) * R_d * y**2 * B_term
    v_gas_sq = G * M_HI * (1 - np.exp(-r_safe/R_d) * (1 + r_safe/R_d)) / r_safe
    return np.sqrt(np.abs(v_disk_sq + v_gas_sq))

def velocity_fluid_MCDM(r, R_gal, V_fluid_max):
    r_safe = np.where(r == 0, 1e-10, r)
    # Pseudo-isothermal profile for flat rotation
    term = 1 - (R_gal / r_safe) * np.arctan(r_safe / R_gal)
    v_sq = V_fluid_max**2 * term
    return np.sqrt(np.abs(v_sq))

def velocity_total_MCDM(r, R_gal, V_fluid_max):
    v_bar = velocity_baryonic(r)
    v_fluid = velocity_fluid_MCDM(r, R_gal, V_fluid_max)
    return np.sqrt(v_bar**2 + v_fluid**2)

# 4. FITTING PROCEDURE
print("Running fit...")
p0_guess = [2.0, 100.0]

try:
    popt, pcov = curve_fit(velocity_total_MCDM, r_data, v_obs,
                           p0=p0_guess, sigma=v_err, absolute_sigma=True)

    R_best, V_best = popt
    perr = np.sqrt(np.diag(pcov))

    # Chi-Square Calculation
    v_model = velocity_total_MCDM(r_data, *popt)
    chi2 = np.sum(((v_obs - v_model) / v_err)**2)
    dof = len(r_data) - 2
    chi2_red = chi2 / dof

    print("="*40)
    print("MCDM FIT RESULTS (NGC 6503)")
    print("="*40)
    print(f"R_gal (Vortex Core)     : {R_best:.2f} +/- {perr[0]:.2f} kpc")
    print(f"V_fluid (Asymptotic)    : {V_best:.2f} +/- {perr[1]:.2f} km/s")
    print(f"Reduced Chi-Square      : {chi2_red:.3f}")
    print("="*40)

    # 5. PLOTTING
    plt.figure(figsize=(10, 6))

    # Plot Data points
    plt.errorbar(r_data, v_obs, yerr=v_err, fmt='ko', label='Observed Data (Begeman 1991)')

    # Smooth lines for models
    r_plot = np.linspace(0.1, 22, 100)

    plt.plot(r_plot, velocity_total_MCDM(r_plot, *popt), 'r-', linewidth=2.5,
             label=f'MCDM Total Fit ($\\chi^2_{{red}}={chi2_red:.3f}$)')
    plt.plot(r_plot, velocity_baryonic(r_plot), 'b--', linewidth=1.5,
             label='Baryons Only (Stars + Gas)')
    plt.plot(r_plot, velocity_fluid_MCDM(r_plot, *popt), 'g:', linewidth=1.5,
             label='MCDM Fluid Vortex')

    plt.title("Galaxy NGC 6503: MCDM Vortex Model Validation", fontsize=14)
    plt.xlabel("Radius (kpc)", fontsize=12)
    plt.ylabel("Rotation Velocity (km/s)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Save AND Show (This ensures you get both the file and the view)
    plt.savefig("NGC6503_MCDM_Fit.png", dpi=300)
    print("Plot saved as 'NGC6503_MCDM_Fit.png'")
    plt.show()

except Exception as e:
    print(f"Error: {e}")
