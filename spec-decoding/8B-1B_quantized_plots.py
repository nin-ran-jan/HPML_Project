import os, pandas as pd, matplotlib.pyplot as plt

data = [
    dict(cfg="8-bit draft", tp=12.818, lat=78.018,  speed=0.834, acc=0.602, util=50.26),
    dict(cfg="4-bit draft", tp=14.423, lat=69.335,  speed=0.938, acc=0.539, util=67.08),
    dict(cfg="8-bit target", tp=14.661, lat=68.207,  speed=0.954, acc=0.601, util=46.25),
    dict(cfg="8-bit tgt + 4-bit drft", tp=11.025, lat=90.702,  speed=0.717, acc=0.537, util=41.54),
]

df = pd.DataFrame(data)

out_dir = "8B-1B_quant_plots"
os.makedirs(out_dir, exist_ok=True)

def plot_metric(metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df.cfg, df[metric], color="tab:blue")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (k=3, thr=0.2, 8B→1B)")
    ax.set_xticklabels(df.cfg, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()

plot_metric("tp", "tokens / s", "throughput.png")
plot_metric("lat", "latency (ms / tok)", "latency.png")
plot_metric("speed", "speed-up × baseline", "speedup.png")
plot_metric("acc", "accept-rate", "accept_rate.png")
plot_metric("util", "GPU util (%)","gpu_util.png")
