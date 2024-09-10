# %%
import glob
import itertools
import os
import pathlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, current_process
from pathlib import Path

import click
import numpy as np
import openap

import pandas as pd
from openap import top
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %%
root_dir = pathlib.Path().resolve()


# %%


def optimize_one(ac, m, d, m_mtow):

    mass = int(round(m * m_mtow, -1))  # rounding up to 10 kg
    dist = int(round(d, -1))  # rounding up to 10 km

    # starting at (0, -150), fly east at equator
    start_lon = -150
    start = (0, start_lon)
    end = (0, start_lon + d / 111.321)

    optimizer = top.CompleteFlight(ac, start, end, m, use_synonym=True)

    n_nodes = int(d / 50)
    n_nodes = min(120, max(30, n_nodes))

    optimizer.setup_dc(nodes=n_nodes)
    # optimizer.debug = True

    flight = optimizer.trajectory(objective="fuel")

    if flight is None:
        return None
    else:
        return (
            flight.drop(["latitude", "longitude", "h"], axis=1)
            .assign(
                x=0,
                y=lambda d: (d.y / 1000).round(-1).astype(int),
                altitude=lambda d: d.altitude.round(-2).astype(int),
                tas=lambda d: d.tas.round().astype(int),
                vertical_rate=lambda d: d.vertical_rate.round().astype(int),
                heading=lambda d: d.heading.round().astype(int),
                mass=lambda d: d.mass.round(-1).astype(int),
                fuel=lambda d: d.fuel.round(-1).astype(int),
            )
            # .assign(yp=lambda d: d.y - d.y.iloc[0])
            # .rename(columns={"yp": "y", "xp": "x"})
            .assign(takeoff_mass=mass)
            .assign(flight_distance=dist)
        )


# %%
def generate_ac(ac, workers=8, grid_size=40, overwrite=False):
    wrap = openap.WRAP(ac, use_synonym=True)
    cruise_range = wrap.cruise_range()
    dmin, dmax = 500, cruise_range["maximum"]

    aircraft = openap.prop.aircraft(ac)
    m_oew = aircraft["limits"]["OEW"]
    m_mtow = aircraft["limits"]["MTOW"]
    mmin, mmax = (m_oew / m_mtow) * 1.2, 0.99

    options = itertools.product(
        np.linspace(mmin, mmax, grid_size), np.linspace(dmin, dmax, grid_size)
    )

    options = pd.DataFrame(options, columns=["mass", "distance"])
    Path(f"{root_dir}/data/optimal/raw/").mkdir(exist_ok=True)
    fout = f"{root_dir}/data/optimal/raw/{ac}.csv"

    if not overwrite and Path(fout).exists():
        return

    flights = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = {
            executor.submit(optimize_one, ac, m, d, m_mtow)
            for i, (m, d) in options.iterrows()
        }
        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            ncols=0,
            desc=f"generating: {ac}",
        ):
            fr = future.result()
            flights.append(fr)

    pd.concat(flights, ignore_index=False).to_csv(fout, index=False)


# %%
@click.command()
@click.option("--ac", required=True, help="aircraft type")
@click.option("--workers", default=6)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--grid-size", default=40)
def main(ac, workers, overwrite, grid_size):
    ac = ac.lower()
    if ac == "all":
        all_acs = openap.prop.available_aircraft()

        # if not overwrite:
        #     files = glob.glob(f"{root_dir}/data/optimal/raw/*.csv")
        #     processed = [Path(f).stem for f in files]
        #     all_acs = set(all_acs) - set(processed)

        for ac in all_acs:
            generate_ac(ac, workers=workers, overwrite=overwrite, grid_size=grid_size)
    else:
        generate_ac(ac, workers=workers, overwrite=overwrite, grid_size=grid_size)


# %%
if __name__ == "__main__":
    main()
