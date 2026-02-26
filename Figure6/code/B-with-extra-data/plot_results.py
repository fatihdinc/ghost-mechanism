import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===============================
# Configuration
# ===============================
results_dir = "results"
lr_all = np.array([float(f"{lr:.2e}") for lr in np.logspace(-3,0,10)] )
alpha_lines = 1
rtol = 1e-6

K_all = np.array([1,2,3,4,5,10,30,50,100]).astype(int)
seeds_all = np.arange(10)   # assuming seeds are 0..9
K_to_idx = {K: i for i, K in enumerate(K_all)}
seed_to_idx = {s: i for i, s in enumerate(seeds_all)}

FILENAME_RE = re.compile(
    r"results_seed(?P<seed>\d+)_K(?P<K>\d+)_lr(?P<lr>[0-9\.e\+\-]+)\.npz"
)

# ===============================
# Plotting
# ===============================

files = os.listdir(results_dir)

parsed = []
for fname in files:
    m = FILENAME_RE.match(fname)
    if m:
        parsed.append({
            "seed": int(m.group("seed")),
            "K": int(m.group("K")),
            "lr": float(m.group("lr")),
            "path": os.path.join(results_dir, fname),
        })

if len(parsed) == 0:
    raise RuntimeError("No result files found.")


met = []

for lr_target in lr_all:
    
    final_loss_mat = np.full((len(seeds_all), len(K_all)), np.nan)
    
    
    runs = [
        r for r in parsed
        if np.isclose(r["lr"], lr_target, rtol=rtol)
    ]

    if len(runs) == 0:
        print(f"No data found for lr={lr_target:.2e}")
        continue

    runs.sort(key=lambda r: r["K"])
    Ks = sorted({r["K"] for r in runs})

    # 🔑 index-based color assignment
    cmap = cm.viridis
    color_map = {
        K: cmap(i / (len(Ks) - 1 if len(Ks) > 1 else 1))
        for i, K in enumerate(Ks)
    }

    plt.figure(figsize=(7, 5))
    print(f"{len(runs)} data found for lr={lr_target:.2e}")
    
    for r in runs:
        data = np.load(r["path"])
        loss = data["loss"]
        loss = loss/loss[0]
    
    
        plt.loglog(np.arange(1,loss.shape[0]+1)[::10],
            loss[::10],
            color=color_map[r["K"]],
            alpha=alpha_lines
        )
        
        final_loss = loss[-1]
        i = seed_to_idx[r["seed"]]
        j = K_to_idx[r["K"]]
        final_loss_mat[i, j] = final_loss

    handles = [
        plt.Line2D([0], [0], color=color_map[K], lw=2, label=f"K={K}")
        for K in Ks
    ]
    plt.legend(handles=handles, title="Rank K")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"log–log loss curves (lr = {lr_target:.2e})")
    plt.ylim([1e-7,2.5])
    plt.xlim([10,1e6])
    plt.savefig(f'learning_curves_lr_{lr_target}.pdf')
    plt.show()
    
    met.append(np.nanmean(final_loss_mat < 1e-2,0));
    
    
#%%

cmap = plt.cm.viridis   # you can swap this for plasma, inferno, magma, cividis
colors = cmap(np.linspace(0, 1, len(met)))


for i in range(len(met)):
    plt.plot(K_all,met[i],color = colors[i])
    plt.xscale('log')

plt.xlabel('K')
plt.ylabel('Fraction of learned networks')
plt.savefig('learned_networks_for_distinct_lr_values.pdf')
plt.show()
