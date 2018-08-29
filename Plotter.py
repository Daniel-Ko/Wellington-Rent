import pandas as pd
from matplotlib import pyplot as plt, dates, ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import Preprocess
import DataReader


def main():
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
    sig_feats_plot, sig_feats_names = plot_sigfeats(df.columns, pipeline)

    # Plot normalised features
    scaled_feat_plot = plot_standardised_data(df[sig_feats_names], pipeline)

    # Finally, plot regression
    predicted_prices = pd.DataFrame(predicted, columns=["Predicted Yearly Rent Price"])
    predicted_prices["Year"] = df.index
    # predicted_prices["Year"] = pd.to_datetime(predicted_prices["Year"])
    # predicted_prices.set_index(["Year"], inplace=True)
    predicted_prices["YearStr"] = predicted_prices["Year"].dt.strftime("%Y-%b-%d")
    predicted_prices["yearfloat"] = dates.datestr2num(predicted_prices["YearStr"])

    fig, ax = plt.subplots()
    g = sns.regplot(
        x="yearfloat",
        y="Predicted Yearly Rent Price",
        data=predicted_prices,
        fit_reg=True,
        scatter=True,
        label="Price pa.",
        ax=ax,
    )
    ax.xaxis.set_major_formatter(date_display_from_float)
    ax.tick_params(labelrotation=45)
    # xticks = ax.get_xticks()
    # ax.set_xticklabels([pd.to_datetime(tick).strftime("%Y") for tick in xticks])

    # ticks = [
    #     pd.to_datetime(tick).strftime("%Y") for tick in predicted_prices["yearstr"]
    # ]
    # g.map(plt.plot, "yearfloat", "Predicted Yearly Rent Price", marker="o")
    # g.set(xticks=ticks)
    # for price in predicted_prices.loc[:, "Predicted Yearly Rent Price"]:
    #     print(price / 12)

    sns.residplot(x="yearfloat", y="Predicted Yearly Rent Price", data=predicted_prices)
    plt.show()


def plot_sigfeats(columns, pipeline):
    feat_importances = pipeline.named_steps[
        "feat_select"
    ].estimator_.feature_importances_

    feat_support = pipeline.named_steps["feat_select"].get_support()

    sig_feats = pd.DataFrame(
        feat_support.reshape(-1, len(feat_support)),
        index=["Important feature?"],
        columns=columns,
    )
    sig_feats.loc["Importance score"] = feat_importances

    ax = sig_feats.iloc[1].plot.bar(
        legend=False,
        y="Importance",
        x="Feature",
        title="Feature importance",
        use_index=True,
        stacked=False,
    )
    ax.set(ylabel="Explained Variance")
    ax.set_title("Feature importance")

    import_feat_mask = sig_feats.loc["Important feature?"] == 1.0
    return (ax, (sig_feats.loc[:, import_feat_mask]).columns)


def plot_standardised_data(df, pipeline):
    scaled_data = StandardScaler().fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    scaled_feat_plot = df_scaled.plot(legend=True, title="Standardised Feature Data")
    scaled_feat_plot.set_ylabel("Standardised Values")
    return scaled_feat_plot


@plt.FuncFormatter
def date_display_from_float(datenum, pos):
    return dates.num2date(datenum).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
