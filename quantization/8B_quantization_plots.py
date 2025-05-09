import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hum_data = [
    dict(q="4‑bit", sampling="greedy",  acc=0.292),
    dict(q="4‑bit", sampling="sampled", acc=0.317),
    dict(q="8‑bit", sampling="greedy",  acc=0.323),
    dict(q="8‑bit", sampling="sampled", acc=0.329),
    dict(q="16‑bit", sampling="greedy",  acc=0.336),
    dict(q="16‑bit", sampling="sampled", acc=0.329),
]
hum_df = pd.DataFrame(hum_data)

wiki_data = [
    dict(q="4-bit",  tp=21.036, lat=47.538,  ppl=6.242, util=70.37),
    dict(q="8-bit",  tp=9.028,  lat=110.765, ppl=5.640, util=48.93),
    dict(q="16-bit", tp=15.499, lat=64.519,  ppl=5.563, util=97.97),
]
wiki_df = (
    pd.DataFrame(wiki_data)
      .set_index("q")
      .loc[["4-bit", "8-bit", "16-bit"]] 
      .reset_index()
)


os.makedirs("8B_humaneval_plots", exist_ok=True)
os.makedirs("8B_wikitext_plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(6,4))
quant_levels = ["4‑bit","8‑bit","16‑bit"]
x = np.arange(len(quant_levels))
width = 0.35

greedy = hum_df[hum_df.sampling=="greedy"].set_index("q").loc[quant_levels]["acc"]
sampled= hum_df[hum_df.sampling=="sampled"].set_index("q").loc[quant_levels]["acc"]

ax.bar(x-width/2, greedy,  width=width, label="greedy" , color="tab:blue")
ax.bar(x+width/2, sampled, width=width, label="sampled", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels(quant_levels)
ax.set_ylabel("accuracy (fraction of 164 tasks)")
ax.set_title("Human‑Eval accuracy by quantization (Llama‑3‑8B)")
ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("8B_humaneval_plots/accuracy_bar.png", dpi=300)
plt.close(fig)

def bar_plot(metric, ylabel, fname, color):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(wiki_df.q, wiki_df[metric], color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by quantization (WikiText‑2, greedy)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("8B_wikitext_plots", fname), dpi=300)
    plt.close(fig)

bar_plot("tp",  "tokens / s", "throughput_bar.png", "tab:green")
bar_plot("lat", "latency (ms / tok)", "latency_bar.png", "tab:red")
bar_plot("ppl", "perplexity", "perplexity_bar.png", "tab:purple")
bar_plot("util", "GPU util (%)", "gpu_util_bar.png", "tab:brown")