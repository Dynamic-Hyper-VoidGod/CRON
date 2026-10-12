
import numpy as np, matplotlib.pyplot as plt
from Reservoir import Reservoir
from readout import Polynomial_Readout
from communication import Communication

# ==================================================
# Configuration
# ==================================================
N = 100
K = 40
WASHOUT_A = 101
WASHOUT_B = 101
TRAIN_FRACTION = 0.8

# ==================================================
# 1. Load NARMA-10 dataset
# ==================================================
dataset = np.loadtxt("narma10.txt", delimiter=",")

x = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1].reshape(-1, 1)

# ==================================================
# 2. Reservoir A: source reservoir
# ==================================================
R_a = Reservoir(
    N=N, alpha=0.5, washout=WASHOUT_A,
    ridge=1e-6, spectral_radius=0.8,
    scale=1, ip_node=1, op_node=1
)

ra = R_a._init_esn(seed=42)
cache_a = R_a._run_res(data=x, reservoir=ra)
states_a = np.asarray(cache_a["states"]).squeeze(-1)

# ==================================================
# 3. Reservoir D: structured disturbance source
# ==================================================
R_d = Reservoir(
    N=N, alpha=0.5, washout=WASHOUT_A,
    ridge=1e-6, spectral_radius=0.8,
    scale=1, ip_node=1, op_node=1
)

rd = R_d._init_esn(seed=44)
cache_d = R_d._run_res(data=x, reservoir=rd)
states_d = np.asarray(cache_d["states"]).squeeze(-1)

if states_a.shape != states_d.shape:
    raise ValueError("A and D state histories are misaligned.")

# ==================================================
# 4. Define target alignment and train/test split
# ==================================================
y_aligned = y[WASHOUT_A + WASHOUT_B:]

expected_samples = len(states_a) - WASHOUT_B

if len(y_aligned) != expected_samples:
    raise ValueError(
        f"Target alignment mismatch: {len(y_aligned)} targets, "
        f"{expected_samples} expected"
    )

split = int(TRAIN_FRACTION * len(y_aligned))

# Centre D using only the interval corresponding to B's
# training inputs. Do not estimate the mean on test data.
d_train_start = WASHOUT_B
d_train_end = d_train_start + split

disturbance_mean = states_d[d_train_start:d_train_end].mean(axis=0)
disturbance = states_d - disturbance_mean

# ==================================================
# 5. Prepare compressed and uncompressed messages
# ==================================================
channel = Communication(K=K, wavelet="haar", level=1)

compressed_states = np.asarray([
    channel.decode(channel.encode(state))
    for state in states_a
])

communication_modes = {
    "uncompressed": states_a.copy(),
    "compressed": compressed_states
}

# ==================================================
# 6. Run all four controlled conditions
# ==================================================
results = {}
predictions = {}

for mode, messages in communication_modes.items():
    for disturbed in (False, True):

        # lambda = 1; disturbance is either enabled or absent.
        if disturbed:
            b_input = messages + disturbance
        else:
            b_input = messages.copy()

        R_b = Reservoir(
            N=N, alpha=0.5, washout=WASHOUT_B,
            ridge=1e-6, spectral_radius=0.8,
            scale=1, ip_node=N, op_node=1
        )

        # Same B initialization across all four conditions.
        rb = R_b._init_esn(seed=43)
        cache_b = R_b._run_res(data=b_input, reservoir=rb)
        states_b = np.asarray(cache_b["states"]).squeeze(-1)

        if len(states_b) != len(y_aligned):
            raise ValueError(
                f"Alignment mismatch in {mode}, disturbed={disturbed}: "
                f"{len(states_b)} states vs {len(y_aligned)} targets"
            )

        x_train, x_test = states_b[:split], states_b[split:]
        y_train, y_test = y_aligned[:split], y_aligned[split:]

        readout_b = Polynomial_Readout()
        readout_b.fit(x_train, y_train)
        y_pred = readout_b.predict(x_test).reshape(-1)

        y_true = y_test.reshape(-1)
        mse = np.mean((y_true - y_pred) ** 2)
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        key = (mode, disturbed)
        results[key] = (mse, correlation)
        predictions[key] = y_pred.copy()

# ==================================================
# 7. Print results
# ==================================================
print("\nCRON: A -> Communication -> B")
print(f"Reservoir size: {N}; K: {K}; lambda: 1.0")
print(f"A states: {states_a.shape}")
print(f"Decoded states: {compressed_states.shape}")
print(f"Training samples: {split}")
print(f"Test samples: {len(y_aligned) - split}")

print("\n{:<16} {:<14} {:>12} {:>14}".format(
    "Communication", "Disturbance", "Test MSE", "Correlation"
))
print("-" * 60)

for mode in ("uncompressed", "compressed"):
    for disturbed in (False, True):
        mse, corr = results[(mode, disturbed)]
        print("{:<16} {:<14} {:>12.6f} {:>14.6f}".format(
            mode,
            "on" if disturbed else "off",
            mse,
            corr
        ))


# ==================================================
# 8. Generate comparison plots
# ==================================================

labels = [
    "Uncompressed\nClean",
    "Compressed\nClean",
    "Uncompressed\nDisturbed",
    "Compressed\nDisturbed"
]

keys = [
    ("uncompressed", False),
    ("compressed", False),
    ("uncompressed", True),
    ("compressed", True)
]

mses = [results[k][0] for k in keys]
correlations = [results[k][1] for k in keys]

# --------------------------------------------------
# Figure 1: Metrics comparison
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(labels, mses)
axes[0].set_title("Held-out MSE")
axes[0].set_ylabel("Mean squared error")
axes[0].tick_params(axis="x", labelsize=9)

axes[1].bar(labels, correlations)
axes[1].set_title("Prediction correlation")
axes[1].set_ylabel("Pearson correlation")
axes[1].set_ylim(-1, 1)
axes[1].axhline(0, linewidth=0.8)
axes[1].tick_params(axis="x", labelsize=9)

fig.suptitle("CRON: Communication and Disturbance Comparison")
fig.tight_layout()
fig.savefig("cron_metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# Figure 2: Target versus predictions
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))

# Display the final 300 points of the held-out test interval.
n_plot = min(300, len(y_test))
target_plot = y_test.reshape(-1)[-n_plot:]
time_axis = np.arange(n_plot)

ax.plot(
    time_axis, target_plot,
    label="True target", linewidth=2
)

for key, label in zip(keys, labels):
    pred_plot = predictions[key][-n_plot:]
    ax.plot(time_axis, pred_plot, label=label.replace("\n", " "))

ax.set_title("CRON: NARMA-10 Target vs Reservoir B Predictions")
ax.set_xlabel("Sample within displayed test interval")
ax.set_ylabel("Target / prediction")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig("cron_prediction_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved plots:")
print("  cron_metrics_comparison.png")
print("  cron_prediction_comparison.png")