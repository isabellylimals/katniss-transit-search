import numpy as np
import matplotlib.pyplot as plt

KIC = "K00753.01"

global_path = f"./data/processed/global_view_{KIC}.npy"
local_path = f"./data/processed/local_view_{KIC}.npy"

global_view = np.load(global_path, allow_pickle=True)

local_view = np.load(local_path, allow_pickle=True)

x_local = np.linspace(-1, 1, local_view.shape[0])

plt.figure(figsize=(12, 4))
plt.plot(global_view, linewidth=0.3)
plt.title(f"GLOBAL VIEW — Folded Light Curve (KIC {KIC})")
plt.xlabel("Sample index")
plt.ylabel("Normalized flux")
plt.grid()
plt.tight_layout()
plt.show()

# -------- LOCAL VIEW --------
plt.figure(figsize=(8, 4))
plt.plot(x_local, local_view, linewidth=1.2)
plt.axvline(0, color="k", ls="--", alpha=0.4)
plt.title(f"LOCAL VIEW — Transit shape (KIC {KIC})")
plt.xlabel("Approx phase (normalized)")
plt.ylabel("Normalized flux (stacked)")
plt.grid()
plt.tight_layout()
plt.show()
