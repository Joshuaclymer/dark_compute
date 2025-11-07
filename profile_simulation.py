#!/usr/bin/env python3
"""Profile the simulation performance"""
import cProfile
import pstats
import io
from model import Model

# Profile Model.run_simulations
pr = cProfile.Profile()
pr.enable()

model = Model(
    year_us_prc_agreement_goes_into_force=2028,
    end_year=2035,
    increment=0.1
)
model.run_simulations(num_simulations=100)

pr.disable()

# Print statistics by total time
print("="*80)
print("TOP FUNCTIONS BY TOTAL TIME:")
print("="*80)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()
ps.sort_stats('tottime')
ps.print_stats(50)
print(s.getvalue())

print("\n" + "="*80)
print("TOP FUNCTIONS BY CUMULATIVE TIME:")
print("="*80)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()
ps.sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
