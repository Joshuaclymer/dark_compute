import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
market_share = [4, 3, 4, 6, 7, 12, 14]

# Define logistic (S-curve) function
def logistic(x, L, k, x0):
    """
    L: maximum value (carrying capacity)
    k: steepness of the curve
    x0: x-value of the sigmoid's midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))

# Convert years to numeric values (years since 2017)
years_numeric = np.array([y - 2017 for y in years])
market_share_array = np.array(market_share)

# Fit the S-curve to the data
# Initial guess: L=100 (max market share), k=0.5, x0=10 (midpoint around 2027)
# Constrain L to be between 20 and 100 (realistic market share bounds)
try:
    popt, pcov = curve_fit(logistic, years_numeric, market_share_array,
                          p0=[80, 0.3, 12], bounds=([20, 0.1, 5], [100, 1.0, 20]), maxfev=10000)
    L_fit, k_fit, x0_fit = popt

    # Generate extrapolation data from 2017 to 2030
    years_extended = np.arange(2017, 2031)
    years_extended_numeric = years_extended - 2017
    market_share_fitted = logistic(years_extended_numeric, L_fit, k_fit, x0_fit)

    # Create the plot
    plt.figure(figsize=(7, 4))

    # Plot historical data as bars
    plt.bar(years, market_share, color='#3498db', alpha=0.7, width=0.6, label='Historical data', zorder=3)

    # Plot fitted S-curve with dotted line
    plt.plot(years_extended, market_share_fitted, color='#e74c3c', linewidth=2,
             linestyle=':', label=f'S-curve fit (L={L_fit:.1f}%)', zorder=2)

    # Add vertical line at 2023 to separate historical from extrapolation
    plt.axvline(x=2023.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    plt.xlabel('Year', fontsize=11)
    plt.ylabel('Chinese SME market share\nin Chinese market (%)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.xlim(2016, 2031)
    plt.ylim(0, max(100, L_fit * 1.1))

    # Add text annotation for extrapolation region
    plt.text(2026.5, 2, 'Extrapolation', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('localization.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to localization.png")
    print(f"\nS-curve parameters:")
    print(f"  Maximum capacity (L): {L_fit:.2f}%")
    print(f"  Growth rate (k): {k_fit:.4f}")
    print(f"  Inflection point: {x0_fit + 2017:.1f}")
    print(f"\nProjected market share for 2030: {logistic(2030-2017, L_fit, k_fit, x0_fit):.1f}%")

except Exception as e:
    print(f"Error fitting S-curve: {e}")
    print("Falling back to simple visualization without extrapolation")

    plt.figure(figsize=(10, 6))
    plt.scatter(years, market_share, color='#3498db', s=100)
    plt.plot(years, market_share, color='#e74c3c', linewidth=2)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Market Share (%)', fontsize=12)
    plt.title('China Semiconductor Equipment Localization Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('localization.png', dpi=300, bbox_inches='tight')
    print("Plot saved to localization.png")