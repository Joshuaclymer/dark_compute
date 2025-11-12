import numpy as np
import matplotlib.pyplot as plt

def survival_probability(years_of_life, initial_hazard_rate, increase_per_year):
    """Calculate survival probability after a given number of years."""
    cumulative_hazard = initial_hazard_rate * years_of_life + increase_per_year * years_of_life**2 / 2
    survival = np.exp(-cumulative_hazard)
    return survival

# Parameters from your model (75th percentile)
initial_hazard_rate = 0.1
increase_per_year = 0.1

print("Multi-Cohort Survival Analysis")
print("=" * 60)
print(f"Initial hazard rate: {initial_hazard_rate}")
print(f"Increase per year: {increase_per_year}")
print()

# Simulate different scenarios of compute addition
scenarios = {
    "All added at start (single cohort)": lambda year: 1000 if year == 0 else 0,
    "Equal amounts added each year": lambda year: 100,
    "Exponential growth (20% per year)": lambda year: 100 * (1.2 ** year),
}

for scenario_name, addition_function in scenarios.items():
    print(f"\n{scenario_name}")
    print("-" * 60)

    # Track compute added per year
    compute_by_year = {}
    max_year = 20

    # Add compute over time
    for year in range(max_year + 1):
        compute_by_year[year] = addition_function(year)

    # Calculate operational and total compute at each point in time
    evaluation_years = [0, 1, 2, 5, 10, 15, 20]

    for eval_year in evaluation_years:
        # Calculate total compute ever added up to this point
        total_compute = sum(compute_by_year[y] for y in range(eval_year + 1))

        # Calculate operational compute (accounting for age-based survival)
        operational_compute = 0
        for install_year in range(eval_year + 1):
            years_of_life = eval_year - install_year
            survival = survival_probability(years_of_life, initial_hazard_rate, increase_per_year)
            operational_compute += compute_by_year[install_year] * survival

        if total_compute > 0:
            survival_rate = operational_compute / total_compute
            print(f"  Year {eval_year:2d}: {survival_rate*100:6.2f}% operational "
                  f"(operational={operational_compute:.1f}, total={total_compute:.1f})")
        else:
            print(f"  Year {eval_year:2d}: No compute yet")

# Now let's create a detailed plot for the exponential growth case
print("\n\nDetailed analysis for exponential growth scenario:")
print("=" * 60)

compute_by_year = {}
for year in range(51):
    compute_by_year[year] = 100 * (1.2 ** year)

evaluation_years = np.arange(0, 51)
survival_rates = []
operational_amounts = []
total_amounts = []

for eval_year in evaluation_years:
    total_compute = sum(compute_by_year[y] for y in range(int(eval_year) + 1))
    operational_compute = 0

    for install_year in range(int(eval_year) + 1):
        years_of_life = eval_year - install_year
        survival = survival_probability(years_of_life, initial_hazard_rate, increase_per_year)
        operational_compute += compute_by_year[install_year] * survival

    if total_compute > 0:
        survival_rate = operational_compute / total_compute
    else:
        survival_rate = 1.0

    survival_rates.append(survival_rate)
    operational_amounts.append(operational_compute)
    total_amounts.append(total_compute)

survival_rates = np.array(survival_rates)
operational_amounts = np.array(operational_amounts)
total_amounts = np.array(total_amounts)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Multi-cohort survival rate
ax = axes[0, 0]
ax.plot(evaluation_years, survival_rates * 100, linewidth=2, label='Multi-cohort survival rate')
ax.set_xlabel('Years Since Start')
ax.set_ylabel('Survival Rate (%)')
ax.set_title('Multi-Cohort Survival Rate\n(Exponential Growth: 20% per year)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Single cohort vs multi-cohort
ax = axes[0, 1]
single_cohort_survival = [survival_probability(y, initial_hazard_rate, increase_per_year) * 100
                          for y in evaluation_years]
ax.plot(evaluation_years, single_cohort_survival, linewidth=2, label='Single cohort', linestyle='--')
ax.plot(evaluation_years, survival_rates * 100, linewidth=2, label='Multi-cohort (exp growth)')
ax.set_xlabel('Years Since Start')
ax.set_ylabel('Survival Rate (%)')
ax.set_title('Single Cohort vs Multi-Cohort Survival')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 100)

# Plot 3: Operational vs Total compute
ax = axes[1, 0]
ax.plot(evaluation_years, total_amounts, linewidth=2, label='Total compute added', alpha=0.7)
ax.plot(evaluation_years, operational_amounts, linewidth=2, label='Operational compute', alpha=0.7)
ax.set_xlabel('Years Since Start')
ax.set_ylabel('Compute (arbitrary units)')
ax.set_title('Operational vs Total Compute Over Time')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_yscale('log')

# Plot 4: Age distribution of operational compute at year 20
ax = axes[1, 1]
eval_year = 20
contributions = []
ages = []
for install_year in range(eval_year + 1):
    years_of_life = eval_year - install_year
    survival = survival_probability(years_of_life, initial_hazard_rate, increase_per_year)
    contribution = compute_by_year[install_year] * survival
    contributions.append(contribution)
    ages.append(years_of_life)

ax.bar(ages, contributions, width=0.8)
ax.set_xlabel('Age of Compute (years)')
ax.set_ylabel('Operational Compute')
ax.set_title(f'Age Distribution of Operational Compute at Year {eval_year}')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multi_cohort_survival_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to multi_cohort_survival_analysis.png")
