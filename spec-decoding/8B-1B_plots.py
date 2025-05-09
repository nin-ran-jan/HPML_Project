import os
import pandas as pd
import matplotlib.pyplot as plt

data = [
    dict(thr=1e-4, sampling=False, k=3, tp=19.995, lat=50.013, speed=1.301, acc=0.547, util=89.08),
    dict(thr=1e-4, sampling=False, k=5, tp=18.653, lat=53.611, speed=1.214, acc=0.440, util=86.79),
    dict(thr=1e-4, sampling=False, k=7, tp=16.696, lat=59.896, speed=1.086, acc=0.363, util=72.30),
    dict(thr=1e-4, sampling=True , k=3, tp=19.833, lat=50.420, speed=1.290, acc=0.549, util=89.88),
    dict(thr=1e-4, sampling=True , k=5, tp=18.226, lat=54.866, speed=1.186, acc=0.431, util=76.08),
    dict(thr=1e-4, sampling=True , k=7, tp=16.510, lat=60.570, speed=1.074, acc=0.361, util=73.84),
    
    dict(thr=0.2, sampling=False, k=3, tp=20.769, lat=48.150, speed=1.351, acc=0.609, util=81.27),
    dict(thr=0.2, sampling=False, k=5, tp=20.535, lat=48.697, speed=1.336, acc=0.544, util=76.91),
    dict(thr=0.2, sampling=False, k=7, tp=20.334, lat=49.178, speed=1.323, acc=0.500, util=77.06),
    dict(thr=0.2, sampling=True , k=3, tp=20.425, lat=48.960, speed=1.329, acc=0.605, util=79.97),
    dict(thr=0.2, sampling=True , k=5, tp=19.385, lat=51.585, speed=1.261, acc=0.490, util=76.61),
    dict(thr=0.2, sampling=True , k=7, tp=19.152, lat=52.215, speed=1.246, acc=0.480, util=75.21),
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
plot_metric("util", "GPU util (%)", "gpu_util.png")
