import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import Preprocess
import DataReader


sns.set()
sns.set_style("whitegrid")

sns.despine()

df = DataReader.create_combined_df()
pipeline = Preprocess.process(df)
regressor = pipeline.named_steps["regressor"]
# print(regressor.mse_path_)


# Score
predicted = pipeline.fit(df, df["Wellington"]).predict(df)
score = pipeline.score(df, df["Wellington"])
print(f"SCORE: {score}")

# Plot significant features
feat_importances = pipeline.named_steps["feat_select"].estimator_.feature_importances_

feat_support = pipeline.named_steps["feat_select"].get_support()

sig_feats = pd.DataFrame(
    feat_support.reshape(-1, len(feat_support)),
    index=["Important feature?"],
    columns=df.columns,
)
sig_feats.loc["Importance score"] = feat_importances

# sig_feats.iloc[1].plot.bar(
#     legend=False,
#     y="Importance",
#     x="Feature",
#     title="Feature importance",
#     use_index=True,
#     stacked=False,
# )

# Plot normalised features
normalised = pipeline.named_steps["standardize"]

scaled_data = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

# df_scaled.plot(legend=False)

predicted_prices = pd.DataFrame(predicted, columns=["Predicted Yearly Rent Price"])
predicted_prices.loc[:, "Year"] = df.index  # df[(df.index > "2007-12-31")]

sns.scatterplot(x="Year", y="Predicted Yearly Rent Price", data=predicted_prices)

for price in predicted_prices.loc[:, "Predicted Yearly Rent Price"]:
    print(price / 12)
# plt.scatter(df.values, df["Wellington"].values, color="black")
# plt.plot(df.index, predicted, color="red", linewidth=3)
plt.show()

# sns.lmplot(x="Year", y=df.columns.values, data=df.values, fit_reg=)
