# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openap
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from openap import top
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

plt.tight_layout()
plt.savefig("figures/ISA_exp.png", bbox_inches="tight", pad_inches=0.1, dpi=200)

plt.show()

# %%

dmin, dmax = 500, max_range
distance = list(range(dmin, int(dmax), 800))
mass = [0.7, 0.75, 0.8, 0.85, 0.9]

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

plt.savefig("figures/alt_vs_dist.png", bbox_inches="tight", pad_inches=0.1, dpi=150)

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

plt.savefig("figures/alt_vs_mass.png", bbox_inches="tight", pad_inches=0.1, dpi=150)

plt.show()

# %%
dataset_real = pd.read_csv("data/a320_estimate_real.csv")
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv")
dataset_opt = dataset_opt.loc[:, ~dataset_opt.columns.str.contains("^Unnamed")]
df_opensky = pd.read_csv("data/a320_estimate_opensky.csv").query(
    "mean_cruise_altitude<40500 and distance<@max_range and takeoff_mass<@m_mtow"
)
df_isa = pd.read_csv("data/optimal/a320_isa_df.csv")
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
    vmin=min(
        c_opt.min(),
        c_opt_max.min(),
        c_real.min(),
        c_pred.min(),
        c_osky.min(),
        c_isa_max.min(),
    ),
    vmax=max(
        c_opt.max(),
        c_opt_max.max(),
        c_real.max(),
        c_pred.max(),
        c_osky.max(),
        c_isa_max.max(),
    ),
)
# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41600, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Mean Cruise Altitude, ft", rotation=0, ha="left")

ax.grid(True)

ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig(
    "figures/lookup.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)


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

ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.savefig(
    "figures/opensky.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
plt.tight_layout()

plt.show()
# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt), s=35)
ax.scatter(
    dataset_real.query("fid== 'ac1-436'").distance.values[0],
    dataset_real.query("fid== 'ac1-436'").mean_cruise_altitude.values[0],
    s=100,
    c="tab:red",
)
ax.scatter(
    dataset_real.query("fid== 'ac1-436'").distance.values[0],
    dataset_real.query("fid== 'ac1-436'").mean_cruise_altitude.values[0],
    c=sm.to_rgba(dataset_real.query("fid== 'ac1-436'").takeoff_mass.values[0] / 1000),
    s=35,
)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41600, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Mean Cruise Altitude, ft", rotation=0, ha="left")

ax.grid(True)

ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig(
    "figures/sample_real_in_lookup.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
plt.show()

# %%
sample_df = pd.read_csv("data/sample_flight/flight.csv")
d = sample_df.distance.max()
start_lon = -150
m = dataset_real.query("fid== 'ac1-436'").pred.values[0] / m_mtow
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
flight.to_csv("data/sample_flight/fuel_optimal_flight.csv")
# %%
sample_df = pd.read_csv("data/sample_flight/flight.csv")
flight = pd.read_csv("data/sample_flight/fuel_optimal_flight.csv")
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

ax.plot(
    sample_df.distance,
    sample_df.altitude,
    color="tab:red",
    label=f"Real, m = {round(sample_df.gw_kg.max()/1000,2)} tons",
)

ax.plot(
    flight.distance,
    flight.altitude,
    color="tab:blue",
    label=f"Optimal, m = {round(flight.mass.max()/1000,2)} tons",
)

ax.set_xlabel("Distance, km")
ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
ax.legend()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)

plt.tight_layout()
plt.savefig(
    f"figures/sample_real_opt_plots_ac1-436.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
plt.show()

# %%
fig = plt.figure(figsize=(11, 4))
gs = matplotlib.gridspec.GridSpec(4, 90)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:4, 0:37])
ax2 = fig.add_subplot(gs[0:4, 43:90])

ax1.scatter(x_real, y_real, c=sm.to_rgba(c_real), s=35)
ax2.scatter(x_pred, y_pred, c=sm.to_rgba(c_pred), s=35)


for ax in [ax1, ax2]:
    ax.grid(True)
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Altitude, ft", rotation=0, ha="left")
    ax.yaxis.set_label_coords(-0.15, 1.02)

ax1.set_title("Real Flights")
ax2.set_title("Estimation using lookup tables")


cbar = plt.colorbar(
    sm,
    ax=ax2,
)
ax1.yaxis.set_label_coords(-0.15, 1.02)
ax2.text(3500, 39700, "TOW, tons")

plt.savefig(
    "figures/real_vs_opt.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
plt.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
ax.hist(dataset_real.error, bins=40, color="tab:blue", edgecolor="gray")
mu1 = dataset_real.error.abs().mean()
median1 = dataset_real.error.median()
sigma1 = dataset_real.error.std()
props = dict(boxstyle="square", facecolor="white", alpha=0.5)
textstr1 = "\n".join(
    (
        r"$\mu_\mathrm{abs}=%.2f$" % (mu1,),
        r"$\mathrm{median}=%.2f$" % (median1,),
        r"$\sigma=%.2f$" % (sigma1,),
    )
)
ax.set_xlabel("Estimation error, kg")
ax.set_ylabel("Flights")
# ax.set_yticklabels([])

# ax.tick_params(axis="y", direction="in")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.text(
    0.6,
    0.95,
    textstr1,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)
# ax.grid(True)
plt.savefig(
    "figures/estimation_err_hist.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
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

ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig(
    "figures/lookup_isa.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)

plt.show()

# %%
plt.figure(figsize=(6, 4))
ax = plt.gca()

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)

ax.scatter(x_opt_max, y_opt_max, c=sm.to_rgba(c_opt_max), s=35)

cbar = plt.colorbar(sm, ax=ax)
ax.text(4600, 41600, "TOW, tons")

ax.set_xlabel("Distance, km")
ax.set_ylabel("Max Cruise Altitude, ft", rotation=0, ha="left")

ax.grid(True)
ax.yaxis.set_label_coords(-0.15, 1.02)

plt.tight_layout()
plt.savefig(
    "figures/lookup_max_cruise.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
plt.show()

# %%
X = dataset_opt[["mean_cruise_altitude", "distance", "mean_cruise_tas"]].copy().values
y = dataset_opt.iloc[:, 4].values

X_real = (
    dataset_real[["mean_cruise_altitude", "distance", "mean_cruise_tas"]].copy().values
)
y_real = dataset_real.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)
df_three_feat = dataset_real
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

diff = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1
)
df_testing = pd.DataFrame(diff, columns=["pred", "test"])
df_testing = df_testing.assign(error=lambda x: x.pred - x.test)

single = regressor.predict(X_real)
df_three_feat = df_three_feat.assign(pred=single)
df_three_feat = df_three_feat.assign(error=lambda x: x.pred - x.takeoff_mass)

df_testing = df_testing.assign(error_percent=lambda x: abs(x.error / x.test) * 100)
df_three_feat = df_three_feat.assign(
    error_percent=lambda x: abs(x.error / x.takeoff_mass) * 100
)
###############
plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.hist(df_three_feat.error, bins=40, edgecolor="gray")
mu1 = df_three_feat.error.abs().mean()
median1 = df_three_feat.error.median()
sigma1 = df_three_feat.error.std()
textstr1 = "\n".join(
    (
        r"$\mu_\mathrm{abs}=%.2f$" % (mu1,),
        r"$\mathrm{median}=%.2f$" % (median1,),
        r"$\sigma=%.2f$" % (sigma1,),
    )
)
ax.set_ylabel("Flights")
ax.set_xlabel("Estimation error, kg")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.set_yticklabels([])
ax.text(
    0.05,
    0.95,
    textstr1,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=props,
)
# ax.grid(True)
plt.savefig(
    "figures/three_feat_estimation_err_hist.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
)
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
        ax.plot(np.array(temp) + i, h, "tab:red", label="ISA model")
        ax.plot(
            np.array(temp_exp) + i,
            h,
            "tab:blue",
            # linewidth=4,
            label="ISA apprixomation",
        )
    else:
        ax.plot(
            np.array(temp) + i,
            h,
            "tab:red",
            linestyle="dashed",
            label="ISA model with shifts" if i == -10 else "_nolegend_",
        )
        ax.plot(
            np.array(temp_exp) + i,
            h,
            "tab:blue",
            linestyle="dashed",
            label="ISA approximation with shifts" if i == -10 else "_nolegend_",
        )
ax.set_ylim(0, 17500)
# ax.grid(True)

ax.set_xlabel("Temperature, K")
ax.set_ylabel("Altitude, m", rotation=0, ha="left")
ax.legend()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.15, 1.02)
plt.tight_layout()
plt.savefig("figures/temperatures.png", bbox_inches="tight", pad_inches=0.1, dpi=200)

plt.show()


# %%
