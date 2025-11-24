import pickle
import numpy as np

# Load the bayesian model data
with open('model_bayesian_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

posterior_samples = model_data['posterior_samples']
A_mean = np.mean(posterior_samples[:, 0])
B_mean = np.mean(posterior_samples[:, 1])
sigma_sq_mean = np.mean(posterior_samples[:, 2])

print(f"y = {A_mean:.3f} / log10(x)^{B_mean:.3f}")
print(f"σ² = {sigma_sq_mean:.3f}")
print(f"n = {len(model_data['X_workers'])}")
