#!/usr/bin/env python3
"""Detailed profiling of simulation performance"""
import cProfile
import pstats
import io
from fab_model import CovertPRCFab, FabModelParameters, USIntelligence, ProcessNode

# Run a single simulation with profiling
pr = cProfile.Profile()
pr.enable()

# Create a single covert fab and detector
covert_fab = CovertPRCFab(
    construction_start_year=2025,
    year_agreement_starts=2028,
    process_node=ProcessNode.nm_28,
    initial_sme_proportion=0.1
)

detector = USIntelligence(
    covert_fab=covert_fab,
    prior_probability_covert_fab_exists=0.1
)

# Run through multiple years to simulate detection
for year in range(2025, 2036):
    year_float = float(year)
    covert_fab.update_year(year_float)
    detector.update_beliefs(year_float)

pr.disable()

# Print detailed statistics
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()
ps.sort_stats('cumulative')
ps.print_stats(50)  # Top 50 functions

print(s.getvalue())

print("\n" + "="*80)
print("BY TOTAL TIME:")
print("="*80)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()
ps.sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())
