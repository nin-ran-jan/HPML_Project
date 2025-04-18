import wandb
import pandas as pd

api = wandb.Api()
run = api.run("ns3888-hpml/final_project/pkw4f7hz")

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