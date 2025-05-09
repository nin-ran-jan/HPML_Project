import wandb
import pandas as pd

# Util file to get Avg GPU utilization from a pariticular run

api = wandb.Api()
# modify run here
run = api.run("ns3888-hpml/final_project/runs/t4cqm9vc")

# Get system metrics history
history = run.history(stream="system")

print("Columns available:")
print(history.columns)

# Access GPU Utilization (%)
if "system.gpu.0.gpu" in history.columns:
    avg_gpu_util = history["system.gpu.0.gpu"].dropna().mean()
    print(f"Avg GPU Utilization: {avg_gpu_util:.4f}%")
else:
    print("GPU Utilization (%) not found in this run.")