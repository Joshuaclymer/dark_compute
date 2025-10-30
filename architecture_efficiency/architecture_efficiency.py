import matplotlib.pyplot as plt
import numpy as np

# H100 baseline (2023)
h100_tpp_per_die_size = 63328
h100_release_year = 2023
annual_efficiency_multiplier = 1.35

def calculate_efficiency(years_since_2023):
    """Calculate TPP per die size based on years since H100 release."""
    return h100_tpp_per_die_size * (annual_efficiency_multiplier ** years_since_2023)

def plot_architecture_efficiency(start_year=2023, end_year=2033):
    """Plot efficiency over time measured by TPP per H100 die size."""
    years = np.arange(start_year, end_year + 1)
    years_since_2023 = years - h100_release_year
    efficiency = [calculate_efficiency(y) for y in years_since_2023]

    plt.figure(figsize=(12, 7))
    plt.plot(years, efficiency, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.fill_between(years, efficiency, alpha=0.3, color='#2E86AB')

    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('TPP per H100 Die Size', fontsize=12, fontweight='bold')
    plt.title('Architecture Efficiency Over Time\n(1.35x annual improvement)',
              fontsize=14, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')

    # Add annotations for key years
    for i, (year, eff) in enumerate(zip(years, efficiency)):
        if i % 2 == 0:  # Annotate every other year to avoid crowding
            plt.annotate(f'{eff:,.0f}',
                        xy=(year, eff),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7)

    plt.tight_layout()
    plt.savefig('architecture_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'architecture_efficiency.png'")

if __name__ == "__main__":
    plot_architecture_efficiency()