# %%
import click
from pathlib import Path
import numpy as np
import openap
import pandas as pd
import glob
from openap import aero
from multiprocessing import Pool

root_dir = Path().resolve()
# %%


def opt_df_creation(ac):
    files = glob.glob(f"data/optimal/{ac}/{ac}*.csv")
    results = []
    for f in files:
        df1 = pd.read_csv(f)
        df = df1.assign(roc=lambda d: d.altitude.diff() / d.ts.diff() * 60)
        df = df.query("altitude>30_000 and -500<roc<500")
        if len(df) == 0:
            continue
        results.append(
            {
                "ac": ac,
                "mean_cruise_altitude": df.altitude.mean(),
                "distance": df1.flight_distance.iloc[0],
                "takeoff_mass": df1.takeoff_mass.iloc[0],
                "landing_mass": df1.mass.min(),
                "mean_cruise_tas": df.tas.mean(),
                "mean_cruise_mach": df.mach.mean(),
            }
        )
    optimal_df = pd.DataFrame.from_dict(results)
    optimal_df.to_csv(f"data/optimal/{ac}_optimal_df.csv")


# %%
@click.command()
@click.option("--ac", required=True, help="aircraft type")
def main(ac):
    ac = ac.lower()
    if ac == "all":
        all_acs = openap.prop.available_aircraft()
        with Pool(8) as pool:
            pool.map(opt_df_creation, all_acs)
    else:
        opt_df_creation(ac)


# %%
if __name__ == "__main__":
    main()
