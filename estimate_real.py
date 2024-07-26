# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
dataset_opt = pd.read_csv("data/optimal/a320_optimal_df.csv")
dataset_opt = dataset_opt.loc[:, ~dataset_opt.columns.str.contains("^Unnamed")]
dataset_opt = dataset_opt.drop(columns=["max_cruise_altitude"])
X = dataset_opt.iloc[:, 1:3].values
y = dataset_opt.iloc[:, 3].values

# %%
dataset_real = pd.read_csv("data/qar_data/a320_real_df.csv")
dataset_real = dataset_real.loc[:, ~dataset_real.columns.str.contains("^Unnamed")]

X_real = dataset_real.iloc[:, 1:3].values
y_real = dataset_real.iloc[:, 3].values
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
# %%
single = regressor.predict(X_real)
dataset_real = dataset_real.assign(pred=single)
dataset_real = dataset_real.assign(error=lambda x: x.pred - x.takeoff_mass)

# %%
df_testing = df_testing.assign(error_percent=lambda x: abs(x.error / x.test) * 100)
dataset_real = dataset_real.assign(
    error_percent=lambda x: abs(x.error / x.takeoff_mass) * 100
)
mean_err = dataset_real.query("mean_cruise_altitude>30000").error_percent.mean()
print(f"Mean estimation error: {round(mean_err,2)}%")
# %%
dataset_real.to_csv("data/a320_estimate_real.csv")
# %%
# Save accuracy check figure
font = 12
plt.figure(figsize=(7, 5))
plt.scatter(df_testing.test / 1000, df_testing.pred / 1000)
plt.plot(
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    [min(df_testing.test / 1000), max(df_testing.pred / 1000)],
    color="red",
    linewidth=2,
    label="error = 0",
)  # y=x line
plt.xlabel("Train Takeoff Mass, tons", fontsize=font)
plt.ylabel("Test Takeoff Mass, tons", fontsize=font)
# plt.title("Model Accuracy check: Train Set vs Test Set", fontsize=font)
plt.legend()
plt.savefig(
    "figures/train_test_acc_check.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=200,
)
plt.show()


# opensky data
# %%
dataset_osky = pd.read_csv("data/opensky/a320_opensky_df.csv")
dataset_osky = dataset_osky.query(
    "distance>800 and mean_cruise_altitude>30_000"
).reset_index(drop=True)
dataset_osky = dataset_osky.loc[:, ~dataset_osky.columns.str.contains("^Unnamed")]
# %%
X_osky = dataset_osky.iloc[:, 1:3].values

# %%
single = regressor.predict(X_osky)
dataset_osky = dataset_osky.assign(takeoff_mass=single)
# %%
dataset_osky.to_csv("data/a320_estimate_opensky.csv")
# %%
