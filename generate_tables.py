# %%
import os
from pathlib import Path
import openap
import pandas as pd

# %%
all_acs = openap.prop.available_aircraft()

# %%
for ac in all_acs:
    if os.path.exists(f"data/optimal/raw/{ac}.csv"):

        df = pd.read_csv(f"data/optimal/raw/{ac}.csv")
        indices = df.index[df["ts"] == 0.0].tolist()

        # Add the first index and the last index of the dataframe
        indices = indices + [len(df)]

        # Create a list to hold individual dataframes
        dfs = []
        Path(f"data/optimal/{ac}").mkdir(exist_ok=True)
        # Iterate over the indices and create dataframes
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]

            flight_df = df.iloc[start_idx:end_idx]
            flight_df = flight_df.reset_index(drop=True)
            mass = flight_df.takeoff_mass[0]
            dist = flight_df.flight_distance[0]
            flight_df.to_csv(
                f"data/optimal/{ac}/{ac}-mass:{mass:06d}-dist:{dist:05d}.csv"
            )
# %%
