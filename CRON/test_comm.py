import numpy as np
from communication import Communication

N = 100
k = 40

state = np.random.default_rng(42).normal(size=N)

channel = Communication(k)
payload = channel.encode(state)
reconstructed = channel.decode(payload)

print("Original shape:", state.shape)
print("Payload shape:", payload.shape)
print("Reconstructed shape:", reconstructed.shape)

relative_error = (np.linalg.norm(state - reconstructed) / (np.linalg.norm(state) + 1e-12))
print("Selected coefficients per band:",channel.selected_lengths)
print("Relative reconstruction error:", relative_error)