import matplotlib.pyplot as plt
from types import SimpleNamespace

from dataset import build_synthetic_lwe_datasets


args = SimpleNamespace(
    num_train=1,
    num_val=1,
    num_test=1,
    n=16,
    m=64,
    q=127,
    h_setting="fixed_h",
    p_nonzero=None,
    fixed_h=2,
    h_min=None,
    h_max=None,
    sigma=1.0,
    noise_distribution="discrete_gaussian",
    noise_bound=None,
    shared_a=False,
    row_permutation="none",
)

train_dataset, _, _ = build_synthetic_lwe_datasets(args, run_seed=42)
item = train_dataset[0]
image = item["image"].squeeze(0).numpy()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plot = ax.imshow(image, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
ax.set_title("Final raw1 input: [A | b] / q")
ax.set_xlabel("n + 1 columns")
ax.set_ylabel("m rows")
fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
