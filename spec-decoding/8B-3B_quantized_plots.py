import os, pandas as pd, matplotlib.pyplot as plt

data = [
    dict(cfg="8-bit draft", tp=9.358,  lat=106.863, speed=0.609, acc=0.663),
    dict(cfg="4-bit draft", tp=10.627, lat=94.103, speed=0.691, acc=0.637),
    dict(cfg="8-bit target", tp=12.804, lat=78.101, speed=0.833, acc=0.661),
    dict(cfg="8-bit tgt + 4-bit drft", tp=8.607, lat=116.188, speed=0.560, acc=0.634),
]

df = pd.DataFrame(data)

out_dir = "8B-3B_quant_plots"
os.makedirs(out_dir, exist_ok=True)

def plot_metric(metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df.cfg, df[metric], color="tab:green")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (k=3, thr=0.2, 8B→3B)")
    ax.set_xticklabels(df.cfg, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()

plot_metric("tp", "tokens / s", "throughput.png")
plot_metric("lat", "latency (ms / tok)", "latency.png")
plot_metric("speed", "speed-up × baseline", "speedup.png")
plot_metric("acc", "accept-rate", "accept_rate.png")
