import wandb
import pandas as pd

# Util file to get Avg GPU util for all runs in the below list

# list of your WandB run paths: "entity/project/run_id"
run_ids = [
    "ns3888-hpml/final_project/plv02j58",
    "ns3888-hpml/final_project/0jij36vh",
    "ns3888-hpml/final_project/3wnazz1f",
    "ns3888-hpml/final_project/epxuzvzc",
    "ns3888-hpml/final_project/wlcjlznr",
    "ns3888-hpml/final_project/x7mzfxgu",
    "ns3888-hpml/final_project/xfc5gp6i",
    "ns3888-hpml/final_project/zrr8j3kx",
    "ns3888-hpml/final_project/bl17amje",
    "ns3888-hpml/final_project/oswyiwrn",
    "ns3888-hpml/final_project/i7dud2hi",
    "ns3888-hpml/final_project/nwfchyoe",
    "ns3888-hpml/final_project/u7zgc76t",
    "ns3888-hpml/final_project/xbh7728h",
    "ns3888-hpml/final_project/8ojnlb7q",
    "ns3888-hpml/final_project/q3y9cea0",
    "ns3888-hpml/final_project/542pr9ap",
    "ns3888-hpml/final_project/40am03zz",
    "ns3888-hpml/final_project/7ruh82hd ",
    "ns3888-hpml/final_project/omc5qofb",
    "ns3888-hpml/final_project/u8qbvcr2",
    "ns3888-hpml/final_project/3wq2r75x",
    "ns3888-hpml/final_project/vi5edms3",
    "ns3888-hpml/final_project/eqc2o6q6",
    "ns3888-hpml/final_project/4yoykx78",
]

# '''
# ns3888-hpml/final_project/4yoykx78
# ns3888-hpml/final_project/eqc2o6q6
# ns3888-hpml/final_project/vi5edms3 8 bit draft
# ns3888-hpml/final_project/3wq2r75x
# ns3888-hpml/final_project/u8qbvcr2
# ns3888-hpml/final_project/omc5qofb 8-3 toks7 0.2
# ns3888-hpml/final_project/7ruh82hd 
# ns3888-hpml/final_project/40am03zz
# ns3888-hpml/final_project/542pr9ap 8-3 toks3 0.2
# ns3888-hpml/final_project/q3y9cea0
# ns3888-hpml/final_project/8ojnlb7q 
# ns3888-hpml/final_project/xbh7728h
# ns3888-hpml/final_project/u7zgc76t 8 bit draft
# '''

api = wandb.Api()
results = []

for run_path in run_ids:
    try:
        run = api.run(run_path)
        history = run.history(stream="system")
        if "system.gpu.0.gpu" in history.columns:
            avg_util = history["system.gpu.0.gpu"].dropna().mean()
            results.append({
                "run_name": run.name,
                "run_id": run.id,
                "gpu_util_avg_%": round(avg_util, 2)
            })
        else:
            results.append({
                "run_name": run.name,
                "run_id": run.id,
                "gpu_util_avg_%": "N/A"
            })
    except Exception as e:
        results.append({
            "run_name": "Error",
            "run_id": run_path,
            "gpu_util_avg_%": f"Error: {e}"
        })

# Create a DataFrame and display results
df = pd.DataFrame(results)
print(df.to_string(index=False))
