# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openap
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from openap import top
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

matplotlib.rc("font", size=12)
matplotlib.rc("font", family="Ubuntu")
matplotlib.rc("lines", linewidth=2, markersize=8)
matplotlib.rc("grid", color="darkgray", linestyle=":")


# %%
ac = "a320"
wrap = openap.WRAP(ac)
aircraft = openap.prop.aircraft(ac)
max_range = wrap.cruise_range()["maximum"]
m_mtow = aircraft["limits"]["MTOW"]
oew = aircraft["limits"]["OEW"]

# %%
# Plotting ISA vs Expoential
h = np.arange(0, 18000, 50)
pressure, density, temp = openap.aero.atmos(h)

a, b, d = [85.46369268, -0.00017235, 213.31449979]
temp_exp = a * np.exp(h * b) + d


fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

ax.plot(temp, h, "tab:blue", label="ISA")
ax.plot(temp_exp, h, "tab:red", label="Exponential approximation")

ax.set_xlabel("Temperature, K")
ax.set_ylabel("Altitude, m", rotation=0, ha="left")
ax.legend()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

plt.tight_layout()
plt.savefig("figures/ISA_exp.png", bbox_inches="tight", dpi=150)

plt.show()

# %%

dmin, dmax = 500, max_range
distance = list(range(dmin, int(dmax), 800))
mass = np.arange(0.7, 0.95, 0.05)

# %%
flights = []

for d in distance:
    start_lon = -150
    start = (0, start_lon)
    end = (0, start_lon + d / 111.321)

    optimizer = top.CompleteFlight(ac, start, end, mass[3])

    flight = optimizer.trajectory(objective="fuel")
    n_nodes = max(30, int(d / 20))
    optimizer.setup_dc(nodes=n_nodes)
    if flight is None:
        continue

    flight = (
        flight.drop(["latitude", "longitude", "h"], axis=1)
        .assign(
            dist=lambda d: ((d.x - d.x.iloc[0]) / 1000).astype(int),
        )
        .drop(["x", "y"], axis=1)
        .assign(flight_id=lambda d: f"{int(round(d.dist.max(),-2))} km")
    )
    flights.append(flight)

flights = pd.concat(flights, ignore_index=True)

flights.to_csv("data/flights_different_distance.csv", index=False)

# %%
flights = pd.read_csv("data/flights_different_distance.csv")

plt.figure(figsize=(6, 4))
ax = plt.gca()
sns.lineplot(
    data=flights, x="dist", y="altitude", hue="flight_id", palette="viridis_r", ax=ax
)
ax.set_xlabel("Distance, km")
ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
ax.legend(loc="right")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

plt.savefig("figures/alt_vs_dist.png", bbox_inches="tight", dpi=150)

plt.show()
# %%
flights = []

for m in mass:
    start_lon = -150
    start = (0, start_lon)
    d = distance[2]
    end = (0, start_lon + d / 111.321)

    optimizer = top.CompleteFlight(ac, start, end, m)

    flight = optimizer.trajectory(objective="fuel")
    n_nodes = max(30, int(d / 20))
    optimizer.setup_dc(nodes=n_nodes)
    if flight is None:
        continue

    flight = (
        flight.drop(["latitude", "longitude", "h"], axis=1)
        .assign(
            dist=lambda d: ((d.x - d.x.iloc[0]) / 1000).astype(int),
        )
        .drop(["x", "y"], axis=1)
        .assign(flight_id=f"{int(m*100)}% MTOW")
    )
    flights.append(flight)

flights = pd.concat(flights, ignore_index=True)

flights.to_csv("data/flights_different_mass.csv", index=False)

# %%

flights = pd.read_csv("data/flights_different_mass.csv")

plt.figure(figsize=(6, 4))
ax = plt.gca()
sns.lineplot(
    data=flights, x="dist", y="altitude", hue="flight_id", palette="viridis_r", ax=ax
)
ax.set_xlabel("Distance, km")
ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
ax.legend()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

plt.savefig("figures/alt_vs_mass.png", bbox_inches="tight", dpi=150)

plt.show()

# %%
dataset_real = pd.read_csv("data/a320_estimate_real.csv")
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv").query("distance>500")
dataset_opt = dataset_opt.loc[:, ~dataset_opt.columns.str.contains("^Unnamed")]
df_opensky = pd.read_csv("data/a320_estimate_opensky.csv").query(
    "mean_cruise_altitude<40500 and distance<@max_range and takeoff_mass<@m_mtow"
)
df_isa = pd.read_csv("data/optimal/a320_isa_df.csv").query("distance>500")
[x_opt, y_opt, c_opt] = [
    dataset_opt.distance,
    dataset_opt.mean_cruise_altitude,
    dataset_opt.takeoff_mass / 1000,
]
[x_opt_max, y_opt_max, c_opt_max] = [
    dataset_opt.distance,
    dataset_opt.max_cruise_altitude,
    dataset_opt.takeoff_mass / 1000,
]
[x_real, y_real, c_real] = [
    dataset_real.distance,
    dataset_real.mean_cruise_altitude,
    dataset_real.takeoff_mass / 1000,
]
[x_pred, y_pred, c_pred] = [
    dataset_real.distance,
    dataset_real.mean_cruise_altitude,
    dataset_real.pred / 1000,
]
[x_osky, y_osky, c_osky] = [
    df_opensky.distance,
    df_opensky.mean_cruise_altitude,
    df_opensky.takeoff_mass / 1000,
]
[x_isa_max, y_isa_max, c_isa_max] = [
    df_isa.distance,
    df_isa.max_cruise_altitude,
    df_isa.takeoff_mass / 1000,
]
norm = Normalize(
    vmin=oew * 1.2 / 1000,
    vmax=m_mtow / 1000,
)

# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41700, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Mean Cruise Altitude, ft", rotation=0, ha="left")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.grid(True)

ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/lookup.png", bbox_inches="tight", dpi=150)

plt.show()

# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_osky, y_osky, c=sm.to_rgba(c_osky), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4500, 40300, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Mean Cruise Altitude, ft", rotation=0, ha="left")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/opensky.png", bbox_inches="tight", dpi=150)
plt.show()


# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt), s=35)
ax.scatter(
    dataset_real.query("fid== 'ac1-571'").distance.values[0],
    dataset_real.query("fid== 'ac1-571'").mean_cruise_altitude.values[0],
    s=100,
    c="tab:red",
    alpha=0.5,
)
ax.scatter(
    dataset_real.query("fid== 'ac1-571'").distance.values[0],
    dataset_real.query("fid== 'ac1-571'").mean_cruise_altitude.values[0],
    c=sm.to_rgba(dataset_real.query("fid== 'ac1-571'").takeoff_mass.values[0] / 1000),
    s=35,
)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41700, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Mean Cruise Altitude, ft", rotation=0, ha="left")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.grid(True)

ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/sample_real_in_lookup.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
sample_df = pd.read_csv("data/sample_flight/flight-ac1-571.csv")
d = sample_df.distance.max()
start_lon = -150
m = dataset_real.query("fid== 'ac1-571'").pred.values[0] / m_mtow
start = (0, start_lon)
end = (0, start_lon + d / 111.321)

optimizer = top.CompleteFlight(ac, start, end, m)

n_nodes = max(30, int(d / 30))
optimizer.setup_dc(nodes=n_nodes)

flight = optimizer.trajectory(objective="fuel")

flight = (
    flight.drop(["latitude", "longitude", "h"], axis=1)
    .assign(
        distance=lambda d: ((d.x - d.x.iloc[0]) / 1000).astype(int),
    )
    .drop(["x", "y"], axis=1)
)
flight.to_csv("data/sample_flight/fuel_optimal_flight-ac1-571.csv")
# %%
sample_df = pd.read_csv("data/sample_flight/flight-ac1-571.csv")
flight = pd.read_csv("data/sample_flight/fuel_optimal_flight-ac1-571.csv")
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

ax.plot(
    sample_df.distance,
    sample_df.altitude,
    color="tab:red",
    label=f"Real, TOW = {round(sample_df.gw_kg.max()/1000,2)} tons",
)

ax.plot(
    flight.distance,
    flight.altitude,
    color="tab:blue",
    label=f"Optimal, TOW = {round(flight.mass.max()/1000,2)} tons",
)

ax.set_xlabel("Distance, km")
ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
ax.legend()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

plt.tight_layout()
plt.savefig(f"figures/sample_real_opt_plots_ac1-571.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
# gs = matplotlib.gridspec.GridSpec(4, 90)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
# ax1 = fig.add_subplot(gs[0:4, 0:37])
# ax2 = fig.add_subplot(gs[0:4, 47:90])

ax1.scatter(x_real, y_real, c=sm.to_rgba(c_real), s=35)
ax2.scatter(x_pred, y_pred, c=sm.to_rgba(c_pred), s=35)


for ax in [ax1, ax2]:
    ax.grid(True)
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
    ax.yaxis.set_label_coords(-0.18, 1.02)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    cbar = plt.colorbar(sm, ax=ax)
    ax.text(3500, 39700, "TOW, tons")

ax1.set_title("Real flights", fontsize=12)
ax2.set_title("Estimation using lookup tables", fontsize=12)
plt.tight_layout()
plt.savefig("figures/real_vs_opt.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
ax.hist(dataset_real.error, bins=40, color="tab:blue", edgecolor="gray")

me = dataset_real.error.mean()
mape = dataset_real.error_percent.mean()
mae = dataset_real.error.abs().mean()
median = dataset_real.error.median()
std = dataset_real.error.std()

textstr1 = f"""\
MAE:  {mae:.2f}
MAPE: {mape:.2f}%
ME:   {me:.2f}
STD:  {std:.2f}\
"""

ax.set_xlabel("Estimation error, kg")
ax.set_ylabel("Flights", rotation=0, ha="left")
ax.yaxis.set_label_coords(-0.095, 1.02)
# ax.set_yticklabels([])

# ax.tick_params(axis="y", direction="in")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.text(
    0.6,
    0.95,
    textstr1,
    transform=ax.transAxes,
    fontsize=14,
    fontfamily="monospace",
    verticalalignment="top",
)
plt.savefig("figures/estimation_err_hist.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
tabl = []
for i in list(range(39, 31, -1)):
    alt = i * 1000
    alt_min = alt - 500
    alt_max = alt + 500

    if i == 39:
        alt_max = 50_000
    elif i == 32:
        alt_min = 0

    me = dataset_real.query("@alt_max > mean_cruise_altitude > @alt_min").error.mean()
    mae = (
        dataset_real.query("@alt_max > mean_cruise_altitude > @alt_min")
        .error.abs()
        .mean()
    )
    mape = dataset_real.query(
        "@alt_max > mean_cruise_altitude > @alt_min"
    ).error_percent.mean()
    num_fli = len(dataset_real.query("@alt_max > mean_cruise_altitude > @alt_min"))
    tabl.append(
        {
            "# of flights": num_fli,
            "alt, ft": alt,
            "me, kg": me.astype(int),
            "mae, kg": mae.astype(int),
            "mape %": mape,
        }
    )
    tabl_df = pd.DataFrame.from_dict(tabl)

tabl_df


# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_isa_max, y_isa_max, c=sm.to_rgba(c_isa_max), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41600, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Max Cruise Altitude, ft", rotation=0, ha="left")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/lookup_isa.png", bbox_inches="tight", dpi=150)

plt.show()

# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_opt_max, y_opt_max, c=sm.to_rgba(c_opt_max), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41750, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Max Cruise Altitude, ft", rotation=0, ha="left")
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)

plt.tight_layout()
plt.savefig("figures/lookup_max_cruise.png", bbox_inches="tight", dpi=150)
plt.show()


# %%
def calculate_error_stats(df, error_cols, mape_cols):

    stats = []
    for err_col, mape_col in zip(error_cols, mape_cols):
        mae = df[err_col].abs().mean()
        me = df[err_col].mean()
        mape = df[mape_col].mean()
        std = df[err_col].std()
        stats.append(
            (
                f"""\
                MAE:  {mae:.2f}
                MAPE: {mape:.2f}%
                ME:   {me:.2f}
                STD:  {std:.2f}\
                """
            )
        )
    return stats


def plot_histograms(df, error_columns, mape_columns, titles, file_name):
    """Plot histograms with error statistics."""
    fig = plt.figure(figsize=(11, 11))
    gs = gridspec.GridSpec(38, 85)
    axes = []

    for i in range(int(len(error_columns) / 2)):
        ax_mult = fig.add_subplot(gs[14 * i : 14 * (i + 1) - 5, 0:40])
        ax_boost = fig.add_subplot(gs[14 * i : 14 * (i + 1) - 5, 45:85])
        axes.extend([ax_mult, ax_boost])

    for ax, col, title in zip(axes, error_columns, titles):

        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        counts, edges, patches = ax.hist(
            df[col], bins=40, edgecolor="gray", color="tab:blue"
        )
        for i, patch in enumerate(patches):
            if edges[i] >= Q1 and edges[i + 1] <= Q3:
                if "mult" in col:
                    c = "sandybrown"
                else:
                    c = "lightskyblue"
                patch.set_facecolor(c)
                patch.set_edgecolor(c)
            else:
                patch.set_facecolor("gray")
                patch.set_edgecolor("gray")
        ax.axvline(0, color=".3", dashes=(2, 2), alpha=0.6)
        ax.axvline(df[col].median(), color="k", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Flights", rotation=0, ha="left")
        ax.yaxis.set_label_coords(-0.095, 1.02)
        ax.set_xlabel("Estimation error, kg")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    stats = calculate_error_stats(df, error_columns, mape_columns)
    for ax, stat in zip(axes, stats):
        ax.text(
            0.15,
            0.95,
            stat,
            transform=ax.transAxes,
            fontsize=11,
            fontfamily="monospace",
            verticalalignment="top",
        )

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.show()


def plot_boxplots(df, error_columns, file_name):
    df_melted = df.melt(
        value_vars=error_columns,
        var_name="Model",
        value_name="Estimation Error",
    )

    df_melted["Model Type"] = df_melted["Model"].apply(
        lambda x: "Multilinear" if "mult" in x else "Boosting"
    )
    df_melted["Feature Set"] = df_melted["Model"].apply(
        lambda x: ("3D" if "_atd" in x else "2D")
    )

    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(
        x="Feature Set",
        y="Estimation Error",
        hue="Model Type",
        palette=["sandybrown", "lightskyblue"],
        data=df_melted,
        gap=0.05,
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Multilinear", "Boosting"])
    ax.axhline(0, color=".3", dashes=(2, 2), alpha=0.6)
    ax.set_ylabel("Error, tons", rotation=0, ha="left")
    ax.yaxis.set_label_coords(-0.095, 1.04)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 1000), ","))
    )
    ax.set(xlabel=None)
    sns.despine(offset=10, trim=True)
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1, dpi=150)

    plt.show()


# %%
y = dataset_opt[["takeoff_mass"]].values
X_ad_real = dataset_real[["mean_cruise_altitude", "distance"]].values
X_ad = dataset_opt[["mean_cruise_altitude", "distance"]].values
X_atd_real = dataset_real[
    ["mean_cruise_altitude", "mean_cruise_tas", "distance"]
].values
X_atd = dataset_opt[["mean_cruise_altitude", "mean_cruise_tas", "distance"]].values
df = dataset_real[
    [
        "fid",
        "mean_cruise_altitude",
        "distance",
        "takeoff_mass",
        "mean_cruise_tas",
    ]
]

# Feature scaling
sc_ad = StandardScaler()
X_ad = sc_ad.fit_transform(X_ad)
X_ad_real = sc_ad.transform(X_ad_real)
sc_atd = StandardScaler()
X_atd = sc_atd.fit_transform(X_atd)
X_atd_real = sc_atd.transform(X_atd_real)
# Set up models and predictions
reg_ad = LinearRegression().fit(X_ad, y)
est_ad = HistGradientBoostingRegressor().fit(X_ad, y)
reg_atd = LinearRegression().fit(X_atd, y)
est_atd = HistGradientBoostingRegressor().fit(X_atd, y)

mult_pred_ad = reg_ad.predict(X_ad_real)
boost_pred_ad = est_ad.predict(X_ad_real)
mult_pred_atd = reg_atd.predict(X_atd_real)
boost_pred_atd = est_atd.predict(X_atd_real)

# Add predictions and errors to dataframe
df = (
    df.assign(
        boost_pred_ad=boost_pred_ad,
        mult_pred_ad=mult_pred_ad,
        boost_pred_atd=boost_pred_atd,
        mult_pred_atd=mult_pred_atd,
    )
    .assign(
        boost_error_ad=lambda x: x.boost_pred_ad - x.takeoff_mass,
        mult_error_ad=lambda x: x.mult_pred_ad - x.takeoff_mass,
        boost_error_atd=lambda x: x.boost_pred_atd - x.takeoff_mass,
        mult_error_atd=lambda x: x.mult_pred_atd - x.takeoff_mass,
    )
    .assign(
        boost_mape_ad=lambda x: abs(x.boost_error_ad / x.takeoff_mass) * 100,
        mult_mape_ad=lambda x: abs(x.mult_error_ad / x.takeoff_mass) * 100,
        boost_mape_atd=lambda x: abs(x.boost_error_atd / x.takeoff_mass) * 100,
        mult_mape_atd=lambda x: abs(x.mult_error_atd / x.takeoff_mass) * 100,
    )
)

error_columns = [
    "mult_error_ad",
    "boost_error_ad",
    "mult_error_atd",
    "boost_error_atd",
]

titles = [
    "Multilinear - 2D",
    "Boosting - 2D",
    "Multilinear - 3D",
    "Boosting - 3D",
]

mape_columns = [
    "mult_mape_ad",
    "boost_mape_ad",
    "mult_mape_atd",
    "boost_mape_atd",
]

plot_histograms(
    df, error_columns, mape_columns, titles, "figures/compare_models_err_hist.png"
)

plot_boxplots(df, error_columns, "figures/compare_models_boxplot.png")

# %%
for err_col, mape_col, title in zip(error_columns, mape_columns, titles):
    stats_tab = []
    for i in list(range(39, 31, -1)):
        alt = i * 1000
        alt_min = alt - 500
        alt_max = alt + 500

        if i == 39:
            alt_max = 50_000
        elif i == 32:
            alt_min = 0

        me = df.query("@alt_max > mean_cruise_altitude > @alt_min")[err_col].mean()
        mae = (
            df.query("@alt_max > mean_cruise_altitude > @alt_min")[err_col].abs().mean()
        )
        mape = df.query("@alt_max > mean_cruise_altitude > @alt_min")[mape_col].mean()
        num_fli = len(df.query("@alt_max > mean_cruise_altitude > @alt_min"))
        stats_tab.append(
            {
                "# of flights": num_fli,
                "alt, ft": alt,
                "me, kg": me.astype(int),
                "mae, kg": mae.astype(int),
                "mape %": mape,
            }
        )
        df_stats = pd.DataFrame.from_dict(stats_tab)
    print(title)
    print(df_stats)

# %%
X = dataset_opt[["mean_cruise_altitude", "distance", "mean_cruise_tas"]].values
y = dataset_opt[["takeoff_mass"]].values

X_real = dataset_real[["mean_cruise_altitude", "distance", "mean_cruise_tas"]].values
y_real = dataset_real[["takeoff_mass"]].values
df_three_feat = dataset_real
regressor = LinearRegression().fit(X, y)
pred = regressor.predict(X_real)
df_three_feat = df_three_feat.assign(pred=pred)
df_three_feat = df_three_feat.assign(error=lambda x: x.pred - x.takeoff_mass)
df_three_feat = df_three_feat.assign(mape=lambda x: abs(x.error / x.takeoff_mass) * 100)

###############
plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.hist(df_three_feat.error, bins=40, edgecolor="gray")

me = df_three_feat.error.mean()
mape = df_three_feat.mape.mean()
mae = df_three_feat.error.abs().mean()
median = df_three_feat.error.median()
std = df_three_feat.error.std()

textstr1 = f"""\
MAE:  {mae:.2f}
MAPE: {mape:.2f}%
ME:   {me:.2f}
STD:  {std:.2f}\
"""

ax.set_ylabel("Flights", rotation=0, ha="left")
ax.yaxis.set_label_coords(-0.095, 1.02)
ax.set_xlabel("Estimation error, kg")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.set_yticklabels([])
ax.text(
    0.65,
    0.95,
    textstr1,
    fontsize=14,
    fontfamily="monospace",
    transform=ax.transAxes,
    verticalalignment="top",
)
plt.savefig("figures/three_feat_estimation_err_hist.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
import xarray as xr

wind = xr.open_dataset(
    f"data/temperature/adaptor.mars.internal-1714739498.5300884-18127-11-2ef1ac2f-16ab-4756-96c1-07f1e9f761d7.grib",
    engine="cfgrib",
)

df_wind = wind.to_dataframe().reset_index().query("time.dt.hour==0")

df_wind = df_wind.assign(
    altitude=lambda x: (1 - (x.isobaricInhPa / 1013.25) ** 0.190284) * 145366.45
)
df_wind = df_wind.query("altitude<60000")

# %%
h = np.arange(0, 18000, 50)
p, rho, T = openap.aero.atmos(h)

a, b, d = [85.46369268, -0.00017235, 213.31449979]
temp_exp = a * np.exp(h * b) + d


# %%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()

for i in range(-90, 90, 10):
    for j in range(-180, 180, 20):
        ax.plot(
            df_wind.query("latitude == @i and longitude == @j").t,
            df_wind.query("latitude == @i and longitude == @j").altitude * 0.3048,
            "wheat",
            label="Real temperature" if i == 0 and j == 0 else "_nolegend_",
        )
for i in range(-20, 30, 10):
    if i == 0:
        ax.plot(temp + i, h, "tab:red", label="ISA model")
        ax.plot(
            temp_exp + i,
            h,
            "tab:blue",
            # linewidth=4,
            label="ISA apprixomation",
        )
    else:
        ax.plot(
            temp + i,
            h,
            "tab:red",
            linestyle="dashed",
            label="ISA model with shifts" if i == -10 else "_nolegend_",
        )
        ax.plot(
            temp_exp + i,
            h,
            "tab:blue",
            linestyle="dashed",
            label="ISA approximation with shifts" if i == -10 else "_nolegend_",
        )
ax.set_ylim(0, 17500)
# ax.grid(True)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
ax.set_xlabel("Temperature, K")
ax.set_ylabel("Altitude, m", rotation=0, ha="left")
ax.legend()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/temperatures.png", bbox_inches="tight", dpi=150)

plt.show()


# %%
