import os
import pandas as pd
import matplotlib.pyplot as plt

data = [
    dict(thr=1e-4, sampling=False, k=3, tp=19.995, lat=1/19.995*1e3, speed=1.301, acc=0.547),
    dict(thr=1e-4, sampling=False, k=5, tp=18.653, lat=1/18.653*1e3, speed=1.214, acc=0.440),
    dict(thr=1e-4, sampling=False, k=7, tp=16.696, lat=1/16.696*1e3, speed=1.086, acc=0.363),
    dict(thr=1e-4, sampling=True , k=3, tp=19.833, lat=1/19.833*1e3, speed=1.290, acc=0.549),
    dict(thr=1e-4, sampling=True , k=5, tp=18.226, lat=1/18.226*1e3, speed=1.186, acc=0.431),
    dict(thr=1e-4, sampling=True , k=7, tp=16.510, lat=1/16.510*1e3, speed=1.074, acc=0.361),
    
    dict(thr=0.2, sampling=False, k=3, tp=20.769, lat=1/20.769*1e3, speed=1.351, acc=0.609),
    dict(thr=0.2, sampling=False, k=5, tp=20.535, lat=1/20.535*1e3, speed=1.336, acc=0.544),
    dict(thr=0.2, sampling=False, k=7, tp=20.334, lat=1/20.334*1e3, speed=1.323, acc=0.500),
    dict(thr=0.2, sampling=True , k=3, tp=20.425, lat=1/20.425*1e3, speed=1.329, acc=0.605),
    dict(thr=0.2, sampling=True , k=5, tp=19.385, lat=1/19.385*1e3, speed=1.261, acc=0.490),
    dict(thr=0.2, sampling=True , k=7, tp=19.152, lat=1/19.152*1e3, speed=1.246, acc=0.480),
]
df = pd.DataFrame(data)

out_dir = "8B-1B_plots"
os.makedirs(out_dir, exist_ok=True)

def plot_metric(metric: str, ylabel: str, fname: str):
    fig, ax = plt.subplots(figsize=(6,4))
    markers = {False: "o", True: "s"}
    for thr in sorted(df["thr"].unique()):
        for samp in [False, True]:
            sub = df[(df["thr"] == thr) & (df["sampling"] == samp)]
            ax.plot(sub["k"], sub[metric],
                    marker=markers[samp],
                    label=f"thr={thr} | {'sample' if samp else 'greedy'}")
    ax.set_xlabel("assist_toks (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs assist_toks (8B→1B)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)

plot_metric("tp", "tokens / s", "throughput.png")
plot_metric("lat", "latency (ms / tok)", "latency.png")
plot_metric("speed", "speed-up × baseline", "speedup.png")
plot_metric("acc", "accept-rate", "accept_rate.png")
