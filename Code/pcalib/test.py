import numpy as np

from pcalib.synthetic import build_corridor_potential
from pcalib.benchmarks import extrapolate_three_series

rng = np.random.default_rng(123)

pot_true = build_corridor_potential(
    T=100,
    n_animals=2,
    neurons_per_animal=20,
    var_array=(4.0, 1.0),
    epsilon_corridor=0.1,
    tau_sigma=1.0,
    tau_xi=7.0,
    rng=rng,
)

result = extrapolate_three_series(
    pot_true,
    K=2,
    gaussian_kernel_width=1.0,
    base_trials_series=[5, 10, 15],
    target_trials=[5, 10, 15, 20],
    mode="trial-averaged",
    gamma=0.05,
    method="diagonal",
)

print("sqrt_mean_sigma shape:", result["sqrt_mean_sigma"].shape)
print("xi shape:", result["xi"].shape)
print("epsilon shape:", result["epsilon"].shape)
print("rho shape:", result["rho"].shape)

print("Any finite epsilon?", np.isfinite(result["epsilon"]).any())
print("Any finite rho?", np.isfinite(result["rho"]).any())
print("Example epsilon:\n", result["epsilon"])
