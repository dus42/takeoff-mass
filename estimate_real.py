# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

matplotlib.rc("font", size=12)
matplotlib.rc("font", family="Ubuntu")
matplotlib.rc("lines", linewidth=2, markersize=8)
matplotlib.rc("grid", color="darkgray", linestyle=":")

# %%
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv")

X = dataset_opt[["mean_cruise_altitude", "distance"]].values
y = dataset_opt[["takeoff_mass"]].values

# %%
dataset_real = pd.read_csv("data/qar_data/a320_real_df.csv")
dataset_real = dataset_real.loc[:, ~dataset_real.columns.str.contains("^Unnamed")]

X_real = dataset_real[["mean_cruise_altitude", "distance"]].values
y_real = dataset_real[["takeoff_mass"]].values
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)
# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
# %%
diff = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1
)
df_testing = pd.DataFrame(diff, columns=["pred", "test"])
df_testing = df_testing.assign(error=lambda x: x.pred - x.test)
df_testing = df_testing.assign(error_percent=lambda x: abs(x.error / x.test) * 100)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Mean test/train estimation error: {round(df_testing.error_percent.mean(),2)}%")
print(f"Mean test/train estimation error: {round(df_testing.error.abs().mean(),2)} kg")
print(f"RMSE: {round(rmse,2)} kg")
print(f"Max test/train estimation error: {round(df_testing.error_percent.max(),2)}%")
print(f"Max test/train estimation error: {round(df_testing.error.max(),2)} kg")
# %%
reg_full = LinearRegression()
reg_full.fit(X, y)
pred = reg_full.predict(X_real)
dataset_real = dataset_real.assign(pred=pred)
dataset_real = dataset_real.assign(error=lambda x: x.pred - x.takeoff_mass)

# %%

dataset_real = dataset_real.assign(
    error_percent=lambda x: abs(x.error / x.takeoff_mass) * 100
)
mean_err = dataset_real.query("mean_cruise_altitude>30000").error_percent.mean()
print(f"Mean estimation error: {round(mean_err,2)}%")
# %%
dataset_real.to_csv("data/a320_estimate_real.csv")
# %%
# Save accuracy check figure
plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.scatter(df_testing.test / 1000, df_testing.pred / 1000, c="tab:blue", s=35)
ax.plot(
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    color="tab:red",
    label="error = 0",
)  # y=x line
ax.set_xlabel("Predicted TOW, tons")
# ax.grid(True)
ax.set_ylabel("Test TOW, tons", rotation=0, ha="left")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.095, 1.02)
ax.legend()
plt.tight_layout()
plt.savefig(
    "figures/train_test_acc_check.png",
    bbox_inches="tight",
    dpi=100,
)
plt.show()

# %%
coefs = list(reg_full.coef_[0])
coefs.append(reg_full.intercept_[0])
print(
    f"Coefficients: TOW = ({round(coefs[1],5)}) * dist + ({round(coefs[0],5)}) * h + ({round(coefs[2],0)})"
)
pd.DataFrame([coefs], columns=["altitude", "distance", "constant"]).to_csv(
    "lin_two_feat_coefs.csv"
)
# opensky data
# %%
dataset_osky = pd.read_csv("data/opensky/a320_opensky_df.csv")
dataset_osky = dataset_osky.query(
    "distance>800 and mean_cruise_altitude>30_000"
).reset_index(drop=True)
dataset_osky = dataset_osky.loc[:, ~dataset_osky.columns.str.contains("^Unnamed")]
# %%
X_osky = dataset_osky[["mean_cruise_altitude", "distance"]].values

# %%
pred = regressor.predict(X_osky)
dataset_osky = dataset_osky.assign(takeoff_mass=pred)
# %%
dataset_osky.to_csv("data/a320_estimate_opensky.csv")


# %%
######################         3D BOOST                #####################
# %%
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv")
dataset_opt = dataset_opt.drop(columns=["max_cruise_altitude"]).query("distance>500")
X = dataset_opt[["mean_cruise_altitude", "mean_cruise_tas", "distance"]].values
y = dataset_opt[["takeoff_mass"]].values

# %%
dataset_real = pd.read_csv("data/qar_data/a320_real_df.csv")
dataset_real = dataset_real.loc[:, ~dataset_real.columns.str.contains("^Unnamed")]

X_real = dataset_real[["mean_cruise_altitude", "mean_cruise_tas", "distance"]].values
y_real = dataset_real[["takeoff_mass"]].values
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)
# %%
regressor = HistGradientBoostingRegressor()
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
# %%
diff = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1
)
df_testing = pd.DataFrame(diff, columns=["pred", "test"])
df_testing = df_testing.assign(error=lambda x: x.pred - x.test)
df_testing = df_testing.assign(error_percent=lambda x: abs(x.error / x.test) * 100)
print(f"Mean test/train estimation error: {round(df_testing.error_percent.mean(),2)}%")
print(f"Max test/train estimation error: {round(df_testing.error_percent.max(),2)}%")
print(f"Max test/train estimation error: {round(df_testing.error.max(),2)} kg")
# %%
reg_full = HistGradientBoostingRegressor()
reg_full.fit(X, y)
pred = reg_full.predict(X_real)
dataset_real = dataset_real.assign(pred=pred)
dataset_real = dataset_real.assign(error=lambda x: x.pred - x.takeoff_mass)

# %%

dataset_real = dataset_real.assign(
    error_percent=lambda x: abs(x.error / x.takeoff_mass) * 100
)
mean_err = dataset_real.query("mean_cruise_altitude>30000").error_percent.mean()
print(f"Mean estimation error: {round(mean_err,2)}%")

# %%
# Save accuracy check figure
plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.scatter(df_testing.test / 1000, df_testing.pred / 1000, c="tab:blue", s=35)
ax.plot(
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    color="tab:red",
    label="error = 0",
)  # y=x line
ax.set_xlabel("Predicted TOW, tons")
# ax.grid(True)
ax.set_ylabel("Test TOW, tons", rotation=0, ha="left")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_label_coords(-0.095, 1.02)
ax.legend()
plt.tight_layout()
plt.savefig(
    "figures/train_test_boost.png",
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()

# %%
