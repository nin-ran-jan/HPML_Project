# uses our raw data from experiments to plot the metrics for 8B-3B spec decoding
import os
import pandas as pd
import matplotlib.pyplot as plt

data = [
    dict(sampling=False, k=3, tp=17.099, lat=58.481, speed=1.112, acc=0.673, util=88.14),
    dict(sampling=False, k=5, tp=16.580, lat=60.313, speed=1.079, acc=0.610, util=92.56),
    dict(sampling=False, k=7, tp=15.928, lat=62.782, speed=1.036, acc=0.559, util=92.61),
    dict(sampling=True , k=3, tp=16.794, lat=59.546, speed=1.093, acc=0.660, util=90.80),
    dict(sampling=True , k=5, tp=16.324, lat=61.261, speed=1.062, acc=0.605, util=91.41),
    dict(sampling=True , k=7, tp=15.475, lat=64.621, speed=1.010, acc=0.556, util=90.45),
]
df = pd.DataFrame(data)

out_dir = "8B-3B_plots"
os.makedirs(out_dir, exist_ok=True)

def plot_metric(metric: str, ylabel: str, fname: str):
    fig, ax = plt.subplots(figsize=(6,4))
    markers = {False: "o", True: "s"}
    for samp in [False, True]:
        sub = df[df["sampling"] == samp]
        ax.plot(sub["k"], sub[metric],
                marker=markers[samp],
                label="sample" if samp else "greedy")
    ax.set_xlabel("assist_toks (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs assist_toks (thr=0.2, 8B→3B)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)

plot_metric("tp", "tokens / s", "throughput.png")
plot_metric("lat", "latency (ms / tok)", "latency.png")
plot_metric("speed", "speed-up × baseline", "speedup.png")
plot_metric("acc", "accept-rate", "accept_rate.png")
plot_metric("util", "GPU util (%)","gpu_util.png")
