import numpy as np
import matplotlib.pyplot as plt

def survival_probability(years_of_life, initial_hazard_rate, increase_per_year):
    """
    Calculate survival probability after a given number of years.

    Args:
        years_of_life: Time since installation (years)
        initial_hazard_rate: Starting hazard rate
        increase_per_year: How much hazard rate increases each year

    Returns:
        Probability that the chip has survived
    """
    # Cumulative hazard: integral of hazard rate from 0 to years_of_life
    # hazard_rate(t) = initial_hazard_rate + increase_per_year * t
    # cumulative_hazard = integral_0^T (initial + increase * t) dt
    #                   = initial*T + increase*T^2/2
    cumulative_hazard = initial_hazard_rate * years_of_life + increase_per_year * years_of_life**2 / 2

    # Survival probability = exp(-cumulative_hazard)
    survival = np.exp(-cumulative_hazard)

    return survival, cumulative_hazard

# Parameters from your model (75th percentile)
initial_hazard_rate_p50 = 0.01
increase_per_year_p50 = 0.01
hazard_rate_p75_multiplier = 10

# Calculate 75th percentile values
initial_hazard_p75 = initial_hazard_rate_p50 * hazard_rate_p75_multiplier
increase_per_year_p75 = increase_per_year_p50 * hazard_rate_p75_multiplier

print(f"75th Percentile Parameters:")
print(f"  Initial hazard rate: {initial_hazard_p75}")
print(f"  Increase per year: {increase_per_year_p75}")
print(f"  After 10 years, hazard rate = {initial_hazard_p75 + 10 * increase_per_year_p75}")
print()

# Calculate survival over time
years = np.linspace(0, 30, 1000)
survivals = []
cumulative_hazards = []

for year in years:
    surv, cum_haz = survival_probability(year, initial_hazard_p75, increase_per_year_p75)
    survivals.append(surv)
    cumulative_hazards.append(cum_haz)

survivals = np.array(survivals)
cumulative_hazards = np.array(cumulative_hazards)

# Find when 80% have burned out (20% survival)
idx_20_percent = np.where(survivals <= 0.20)[0]
if len(idx_20_percent) > 0:
    time_80_percent_burnout = years[idx_20_percent[0]]
    print(f"Time for 80% to burn out (20% survival): {time_80_percent_burnout:.2f} years")
else:
    print("80% burnout not reached in 30 years")

# Find when 99% have burned out (1% survival)
idx_1_percent = np.where(survivals <= 0.01)[0]
if len(idx_1_percent) > 0:
    time_99_percent_burnout = years[idx_1_percent[0]]
    print(f"Time for 99% to burn out (1% survival): {time_99_percent_burnout:.2f} years")
else:
    print("99% burnout not reached in 30 years")

# Find when 99.9% have burned out (0.1% survival)
idx_0_1_percent = np.where(survivals <= 0.001)[0]
if len(idx_0_1_percent) > 0:
    time_99_9_percent_burnout = years[idx_0_1_percent[0]]
    print(f"Time for 99.9% to burn out (0.1% survival): {time_99_9_percent_burnout:.2f} years")
else:
    print("99.9% burnout not reached in 30 years")

print()
print("Survival at key timepoints:")
for year in [1, 2, 5, 10, 15, 20]:
    surv, cum_haz = survival_probability(year, initial_hazard_p75, increase_per_year_p75)
    current_hazard = initial_hazard_p75 + increase_per_year_p75 * year
    print(f"  Year {year:2d}: {surv*100:6.2f}% survival, cumulative hazard = {cum_haz:.3f}, current hazard rate = {current_hazard:.2f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Survival probability over time
ax1.plot(years, survivals * 100, linewidth=2)
ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% survival (80% burned out)')
ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='1% survival (99% burned out)')
ax1.set_xlabel('Years of Life')
ax1.set_ylabel('Survival Probability (%)')
ax1.set_title(f'Chip Survival Over Time\n(initial hazard={initial_hazard_p75}, increase={increase_per_year_p75}/year)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 30)

# Plot 2: Hazard rate over time
hazard_rates = initial_hazard_p75 + increase_per_year_p75 * years
ax2.plot(years, hazard_rates, linewidth=2, color='red')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Hazard rate = 1.0')
ax2.set_xlabel('Years of Life')
ax2.set_ylabel('Hazard Rate (per year)')
ax2.set_title('Hazard Rate Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 30)

plt.tight_layout()
plt.savefig('survival_analysis.png', dpi=150, bbox_inches='tight')
print()
print("Visualization saved to survival_analysis.png")
plt.close()
