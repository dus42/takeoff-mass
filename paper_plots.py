# %%
import numpy as np
import pandas as pd
import openap
from openap import top
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# %%
ac = "a320"
wrap = openap.WRAP(ac)
aircraft = openap.prop.aircraft(ac)
max_range = wrap.cruise_range()["maximum"]
m_mtow = aircraft["limits"]["MTOW"]
oew = aircraft["limits"]["OEW"]
font = 12
# %%
# Plotting ISA vs Expoential
h = range(0, 18000, 50)
temp = []
r = []
for i in range(len(h)):
    pi, rhoi, Ti = openap.aero.atmos(h[i])
    temp.append(Ti)
    r.append(rhoi)

a, b, d = [85.46369268, -0.00017235, 213.31449979]
temp_exp = []
for i in range(len(h)):
    temp_exp.append(a * np.exp(h[i] * b) + d)


fig = plt.figure(figsize=(7, 5))
plt.plot(temp, h, "b", label="ISA")
plt.plot(temp_exp, h, "r", label="Exponential approximation")

plt.xlabel("Temperature, K", fontsize=font)
plt.legend()
plt.ylabel("Altitude, m", fontsize=font)
plt.savefig(
    "figures/ISA_exp.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.show()

# %%

dmin, dmax = 500, max_range
distance = list(range(dmin, int(dmax), 800))
mass = [0.7, 0.75, 0.8, 0.85, 0.9]
# %%
c = 0
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
            x=0,
            y=lambda d: (d.y / 1000).round(-1).astype(int),
            alt=lambda d: d.altitude.round(-2).astype(int),
            tas=lambda d: d.tas.round().astype(int),
            vertical_rate=lambda d: d.vertical_rate.round().astype(int),
            heading=lambda d: d.heading.round().astype(int),
            mass=lambda d: d.mass.round(-1).astype(int),
            fuel=lambda d: d.fuel.round(-1).astype(int),
        )
        .assign(y=lambda d: d.y - d.y.iloc[0])
        .rename(
            columns={
                "alt": f"alt{c}",
                "ts": f"ts{c}",
                "mass": f"mass{c}",
                "tas": f"tas{c}",
                "mach": f"mach{c}",
            }
        )
        .drop(["x", "y", "vertical_rate", "altitude", "heading", "fuel"], axis=1)
    )

    r = []
    for i in range(len(flight)):
        ts = flight[f"ts{c}"][i]
        tas = flight[f"tas{c}"][i] * openap.aero.kts
        if i == 0:
            dis = 0
        else:
            dis = dis + (ts - flight[f"ts{c}"][i - 1]) * tas
        r.append(dis / 1000)
    flight = flight.assign(distance=r).rename(columns={"distance": f"dist{c}"})
    if c == 0:
        df = flight
    else:
        df = pd.concat([df, flight], axis=1)
    c = c + 1

# %%
plt.figure(figsize=(7, 5))
cmap = plt.get_cmap("viridis").reversed()
norm = Normalize(
    vmin=50600,
    vmax=70200,
)
sm = ScalarMappable(norm=norm, cmap=cmap)
for i in range(5):
    plt.plot(
        df[f"dist{i}"],
        df[f"alt{i}"],
        linewidth=2,
        label=str(int(df[f"mass{i}"][0] / 780)) + "% MTOW",
        c=sm.to_rgba(df[f"mass{i}"][0]),
    )  # y=x line

plt.xlabel("Distance, km", fontsize=font)
plt.ylabel("Altitude, ft", fontsize=font)
# plt.title("Cruise Altitude Variation with the Change in Takeoff Mass", fontsize=font)
plt.legend()
plt.savefig(
    "figures/alt_vs_mass.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)

plt.show()
# %%
c = 0
for d in distance:
    start_lon = -150
    start = (0, start_lon)
    # d = distance[2]
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
            x=0,
            y=lambda d: (d.y / 1000).round(-1).astype(int),
            alt=lambda d: d.altitude.round(-2).astype(int),
            tas=lambda d: d.tas.round().astype(int),
            vertical_rate=lambda d: d.vertical_rate.round().astype(int),
            heading=lambda d: d.heading.round().astype(int),
            mass=lambda d: d.mass.round(-1).astype(int),
            fuel=lambda d: d.fuel.round(-1).astype(int),
        )
        .assign(y=lambda d: d.y - d.y.iloc[0])
        .rename(
            columns={
                "alt": f"alt{c}",
                "ts": f"ts{c}",
                "mass": f"mass{c}",
                "tas": f"tas{c}",
                "mach": f"mach{c}",
            }
        )
        .drop(["x", "y", "vertical_rate", "heading", "fuel"], axis=1)
    )

    r = []
    for i in range(len(flight)):
        ts = flight[f"ts{c}"][i]
        tas = flight[f"tas{c}"][i] * openap.aero.kts
        if i == 0:
            dis = 0
        else:
            dis = dis + (ts - flight[f"ts{c}"][i - 1]) * tas
        r.append(dis / 1000)
    flight = flight.assign(distance=r).rename(columns={"distance": f"dist{c}"})
    if c == 0:
        df = flight
    else:
        df = pd.concat([df, flight], axis=1)
    c = c + 1

# %%
plt.figure(figsize=(7, 5))
font = 12
cmap = plt.get_cmap("viridis").reversed()
norm = Normalize(
    vmin=100,
    vmax=4000,
)
sm = ScalarMappable(norm=norm, cmap=cmap)
for i in range(5):
    plt.plot(
        df[f"dist{i}"],
        df[f"alt{i}"],
        linewidth=2,
        c=sm.to_rgba(df[f"dist{i}"].max()),
        label=str(int(df[f"dist{i}"].max())) + " km",
    )  # y=x line
plt.xlabel("Distance, km", fontsize=font)
plt.ylabel("Altitude, ft", fontsize=font)
# plt.title("Cruise Altitude Variation with the Change in Flight Distance", fontsize=font)
plt.legend()
plt.savefig(
    "figures/alt_vs_dist.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.show()

# %%
dataset_real = pd.read_csv("data/a320_estimate_real.csv")
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv")
dataset_opt = dataset_opt.loc[:, ~dataset_opt.columns.str.contains("^Unnamed")]
fig = plt.figure(figsize=(7, 6))
gs = matplotlib.gridspec.GridSpec(8, 1)

[x_opt, y_opt, c_opt] = [
    dataset_opt.distance,
    dataset_opt.mean_cruise_altitude / 1000,
    dataset_opt.takeoff_mass / 1000,
]

norm = Normalize(
    vmin=c_opt.min(),
    vmax=c_opt.max(),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:7, 0])

ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))
ax1.grid(True)
ax1.set_xlabel("Distance, km", fontsize=font)
ax1.set_ylabel("Altitude, 1000 ft", fontsize=font)


cbar = plt.colorbar(
    sm,
    ax=ax1,
    location="bottom",
)

cbar.set_label("Takeoff Mass, tons", fontsize=font)
plt.savefig(
    "figures/lookup.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
plt.show()
# %%

df_opensky = pd.read_csv("data/a320_estimate_opensky.csv")
df_opensky = df_opensky.query(
    "mean_cruise_altitude<40500 and distance<@max_range and takeoff_mass<@m_mtow"
)

fig = plt.figure(figsize=(7, 6))
gs = matplotlib.gridspec.GridSpec(8, 1)

[x_opt, y_opt, c_opt] = [
    df_opensky.distance,
    df_opensky.mean_cruise_altitude / 1000,
    df_opensky.takeoff_mass / 1000,
]

norm = Normalize(
    vmin=c_opt.min(),
    vmax=c_opt.max(),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:7, 0])

ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))


# ax.legend()
# ax.set_xlim(500, 4000)
# ax.set_ylim(30, 40)
ax1.grid(True)
ax1.set_xlabel("Distance, km", fontsize=font)
ax1.set_ylabel("Altitude, 1000 ft", fontsize=font)
# ax1.set_title("Location of the Real Trajectory in the Lookup Table", fontsize=font)

cbar = plt.colorbar(
    sm,
    ax=ax1,
    location="bottom",
)

cbar.set_label("Takeoff Mass, tons", fontsize=font)
plt.savefig(
    "figures/opensky.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
plt.show()
# %%

fig = plt.figure(figsize=(7, 6))
gs = matplotlib.gridspec.GridSpec(8, 1)

[x_opt, y_opt, c_opt] = [
    dataset_opt.distance,
    dataset_opt.mean_cruise_altitude / 1000,
    dataset_opt.takeoff_mass / 1000,
]

norm = Normalize(
    vmin=c_opt.min(),
    vmax=c_opt.max(),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:7, 0])

ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))
ax1.scatter(
    dataset_real.query("fid== 'ac1-436'").distance.values[0],
    dataset_real.query("fid== 'ac1-436'").mean_cruise_altitude.values[0] / 1000,
    s=120,
    c="r",
)
ax1.scatter(
    dataset_real.query("fid== 'ac1-436'").distance.values[0],
    dataset_real.query("fid== 'ac1-436'").mean_cruise_altitude.values[0] / 1000,
    c=sm.to_rgba(dataset_real.query("fid== 'ac1-436'").takeoff_mass.values[0] / 1000),
)


# ax.legend()
# ax.set_xlim(500, 4000)
# ax.set_ylim(30, 40)
ax1.grid(True)
ax1.set_xlabel("Distance, km", fontsize=font)
ax1.set_ylabel("Altitude, 1000 ft", fontsize=font)
# ax1.set_title("Location of the Real Trajectory in the Lookup Table", fontsize=font)

cbar = plt.colorbar(
    sm,
    ax=ax1,
    location="bottom",
)

cbar.set_label("Takeoff Mass, tons", fontsize=font)
plt.savefig(
    "figures/sample_real_in_lookup.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
plt.show()

# %%
ddf = pd.read_csv(f"data/sample_flight/flight.csv")
d = ddf.distance.max()
start_lon = -150
m = dataset_real.query("fid== 'ac1-436'").pred.values[0] / m_mtow
start = (0, start_lon)
end = (0, start_lon + d / 111.321)

optimizer = top.CompleteFlight(ac, start, end, m)

n_nodes = max(30, int(d / 30))
optimizer.setup_dc(nodes=n_nodes)
# optimizer.debug = True

flight = optimizer.trajectory(objective="fuel")

flight = flight.drop(["latitude", "longitude", "h"], axis=1).assign(
    x=0,
    y=lambda d: (d.y / 1000).round(-1).astype(int),
    alt=lambda d: d.altitude.round(-2).astype(int),
    tas=lambda d: d.tas.round().astype(int),
    vs=lambda d: d.vertical_rate.round().astype(int),
    heading=lambda d: d.heading.round().astype(int),
    mass=lambda d: d.mass.round(-1).astype(int),
    fuel=lambda d: d.fuel.round(-1).astype(int),
)
r = []
for i in range(len(flight)):
    ts = flight[f"ts"][i]
    tas = flight[f"tas"][i] * openap.aero.kts
    if i == 0:
        dis = 0
    else:
        dis = dis + (ts - flight[f"ts"][i - 1]) * tas
    r.append(dis / 1000)
flight = flight.assign(distance=r)


# %%
plt.figure(figsize=(7, 5))
font = 12

plt.plot(
    ddf.distance,
    ddf.altitude / 1000,
    linewidth=2,
    color="red",
    label=f"Real, m = {round(ddf.gw_kg.max()/1000,2)} tons",
)  # y=x line


plt.plot(
    flight.distance,
    flight.altitude / 1000,
    linewidth=2,
    label=f"Optimal, m = {round(m*m_mtow/1000,2)} tons",
)  # y=x line, km")
plt.xlabel("Distance, km", fontsize=font)
plt.ylabel("Altitude, 1000 ft", fontsize=font)
# plt.title("Real Trajectory vs Optimal Trajectory", fontsize=font)
plt.legend()
plt.savefig(
    f"figures/sample_real_opt_plots_ac1-436.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.show()

# %%
fig = plt.figure(figsize=(14, 5))
gs = matplotlib.gridspec.GridSpec(4, 90)
[x_r, y_r, c_r] = [
    dataset_real.distance,
    dataset_real.mean_cruise_altitude / 1000,
    dataset_real.takeoff_mass / 1000,
]
[x_opt, y_opt, c_opt] = [
    dataset_real.distance,
    dataset_real.mean_cruise_altitude / 1000,
    dataset_real.pred / 1000,
    # dataset_opt.distance,
    # dataset_opt.mean_cruise_altitude / 1000,
    # dataset_opt.takeoff_mass / 1000,
]


norm = Normalize(
    vmin=min(
        c_r.min(),
        c_opt.min(),
    ),
    vmax=max(
        c_r.max(),
        c_opt.max(),
    ),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:4, 0:38])
ax2 = fig.add_subplot(gs[0:4, 43:90])


ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))
ax2.scatter(x_r, y_r, c=sm.to_rgba(c_r))

for ax in [ax1, ax2]:
    ax.grid(True)
    ax.set_xlabel("Flight Distance, km", fontsize=font)
    ax.tick_params(labelsize=font - 2)
ax1.set_ylabel("Mean Cruise Altitude, 1000 ft", fontsize=font)

ax1.set_title("Estimation using lookup tables", fontsize=font)
ax2.set_title("Real Flights", fontsize=font)

cbar = plt.colorbar(
    sm,
    ax=ax2,
    location="right",
)

font_st = matplotlib.font_manager.FontProperties(size=font)
cbar.ax.yaxis.label.set_font_properties(font_st)
cbar.ax.tick_params(labelsize=font)
cbar.set_label("Takeoff Mass, tons", fontsize=font)

plt.savefig(
    "figures/real_vs_opt.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()
ax1.hist(dataset_real.error, bins=60)
mu1 = dataset_real.error.abs().mean()
median1 = dataset_real.error.median()
sigma1 = dataset_real.error.std()
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
textstr1 = "\n".join(
    (
        r"$\mu_\mathrm{abs}=%.2f$" % (mu1,),
        r"$\mathrm{median}=%.2f$" % (median1,),
        r"$\sigma=%.2f$" % (sigma1,),
    )
)
ax1.set_ylabel("Number of flights", fontsize=font)
ax1.set_xlabel("Estimation error, kg", fontsize=font)
ax1.text(
    0.55,
    0.95,
    textstr1,
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)
ax1.grid(True)
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
        me = dataset_real.query("mean_cruise_altitude > @alt_min").error.mean()
        mae = dataset_real.query("mean_cruise_altitude > @alt_min").error.abs().mean()
        mape = dataset_real.query(
            "mean_cruise_altitude > @alt_min"
        ).error_percent.mean()
        num_fli = len(dataset_real.query("mean_cruise_altitude > @alt_min"))
    elif i == 32:
        me = dataset_real.query("@alt_max > mean_cruise_altitude").error.mean()
        mae = dataset_real.query("@alt_max > mean_cruise_altitude").error.abs().mean()
        mape = dataset_real.query(
            "@alt_max > mean_cruise_altitude"
        ).error_percent.mean()
        num_fli = len(dataset_real.query("@alt_max > mean_cruise_altitude"))
    else:
        me = dataset_real.query(
            "@alt_max > mean_cruise_altitude > @alt_min"
        ).error.mean()
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
            "flights": num_fli,
            "alt": alt,
            "me": me,
            "mae": mae,
            "mape": mape,
        }
    )
    tabl_df = pd.DataFrame.from_dict(tabl)

tabl_df

# %%
df_isa = pd.read_csv("data/optimal/a320_isa_df.csv")
fig = plt.figure(figsize=(7, 6))
gs = matplotlib.gridspec.GridSpec(8, 1)

[x_opt, y_opt, c_opt] = [
    df_isa.distance,
    df_isa.max_cruise_altitude / 1000,
    df_isa.takeoff_mass / 1000,
]

norm = Normalize(
    vmin=c_opt.min(),
    vmax=c_opt.max(),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:7, 0])

ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))
ax1.grid(True)
ax1.set_xlabel("Distance, km", fontsize=font)
ax1.set_ylabel("Altitude, 1000 ft", fontsize=font)


cbar = plt.colorbar(
    sm,
    ax=ax1,
    location="bottom",
)

cbar.set_label("Takeoff Mass, tons", fontsize=font)
plt.savefig(
    "figures/lookup_isa.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(7, 6))
gs = matplotlib.gridspec.GridSpec(8, 1)

[x_opt, y_opt, c_opt] = [
    dataset_opt.distance,
    dataset_opt.max_cruise_altitude / 1000,
    dataset_opt.takeoff_mass / 1000,
]

norm = Normalize(
    vmin=c_opt.min(),
    vmax=c_opt.max(),
)

cmap = plt.get_cmap("viridis").reversed()
sm = ScalarMappable(norm=norm, cmap=cmap)
ax1 = fig.add_subplot(gs[0:7, 0])

ax1.scatter(x_opt, y_opt, c=sm.to_rgba(c_opt))
ax1.grid(True)
ax1.set_xlabel("Distance, km", fontsize=font)
ax1.set_ylabel("Altitude, 1000 ft", fontsize=font)


cbar = plt.colorbar(
    sm,
    ax=ax1,
    location="bottom",
)

cbar.set_label("Takeoff Mass, tons", fontsize=font)
plt.savefig(
    "figures/lookup_isa.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.tight_layout()
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
dataset_real = dataset_real.assign(pred=single)
dataset_real = dataset_real.assign(error=lambda x: x.pred - x.takeoff_mass)

df_testing = df_testing.assign(error_percent=lambda x: abs(x.error / x.test) * 100)
dataset_real = dataset_real.assign(
    error_percent=lambda x: abs(x.error / x.takeoff_mass) * 100
)

# %%
dataset_real = dataset_real.query("landing_mass>0")
font = 12
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()
ax1.hist(dataset_real.error, bins=60)
mu1 = dataset_real.error.abs().mean()
median1 = dataset_real.error.median()
sigma1 = dataset_real.error.std()
textstr1 = "\n".join(
    (
        r"$\mu_\mathrm{abs}=%.2f$" % (mu1,),
        r"$\mathrm{median}=%.2f$" % (median1,),
        r"$\sigma=%.2f$" % (sigma1,),
    )
)
ax1.set_ylabel("Number of flights", fontsize=font)
ax1.set_xlabel("Estimation error, kg", fontsize=font)
ax1.text(
    0.05,
    0.95,
    textstr1,
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)
ax1.grid(True)
plt.savefig(
    "figures/three_feat_estimation_err_hist.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
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
h = range(0, 18000, 50)
temp = []
r = []
for i in range(len(h)):
    pi, rhoi, Ti = openap.aero.atmos(h[i])
    temp.append(Ti)

a, b, d = [85.46369268, -0.00017235, 213.31449979]
temp_exp = []
for i in range(len(h)):
    temp_exp.append(a * np.exp(h[i] * b) + d)


# %%
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

for i in range(-90, 90, 10):
    for j in range(-180, 180, 20):

        ax.plot(
            df_wind.query("latitude == @i and longitude == @j").t,
            df_wind.query("latitude == @i and longitude == @j").altitude * 0.3048,
            "-g",
            label="Real temperature" if i == 0 and j == 0 else "_nolegend_",
        )
for i in range(-20, 30, 10):
    if i == 0:
        ax.plot(np.array(temp) + i, h, "-r", linewidth=4, label="ISA model")
        ax.plot(np.array(temp_exp) + i, h, "-b", linewidth=4, label="ISA apprixomation")
    else:
        ax.plot(
            np.array(temp) + i,
            h,
            "-r",
            label="ISA model with shifts" if i == -10 else "_nolegend_",
        )
        ax.plot(
            np.array(temp_exp) + i,
            h,
            "-b",
            label="ISA approximation with shifts" if i == -10 else "_nolegend_",
        )
ax.set_ylim(0, 18000)
ax.grid(True)
ax.legend()
ax.set_ylabel("Altitude, m", fontsize=font)
ax.set_xlabel("Temperatuer, K", fontsize=font)
plt.savefig(
    "figures/temperatures.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.show()
# %%
