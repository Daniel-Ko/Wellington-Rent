import argparse

import pandas as pd
from matplotlib import pyplot as plt, dates, ticker
import seaborn as sns
import statsmodels
import patsy

from sklearn.preprocessing import StandardScaler

import Preprocess
import DataReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--add_region", action="store_true")
    parser.add_argument("-t", "--add_TA", action="store_true")
    args = parser.parse_args()
    
    sns.set()
    sns.set_style("whitegrid")

    sns.despine()

    df = DataReader.create_combined_df(args.add_region, args.add_TA)

    pipeline = Preprocess.process(df)
    # regressor = pipeline.named_steps["regressor"]
    # print(regressor.mse_path_)

    # Score
    test_data = DataReader.get_test_data()["Wellington"]
    predicted = pipeline.fit(df, test_data).predict(df)
    score = pipeline.score(df, test_data)
    print(f"SCORE: {score}")

    # Plot significant features
    sig_feats_plot, sig_feats_names = plot_sigfeats(df.columns, pipeline)

    # Plot normalised features
    scaled_feat_plot = plot_standardised_data(df[sig_feats_names], pipeline)

    # Finally, plot regression
    regplot, predicted_prices = plot_regression(df, predicted)

    # And residuals to check the fit of the regression
    residplot = plot_reg_resid(predicted_prices, test_data)

    for price in predicted_prices.loc[:, "Predicted Yearly Rent Price"]:
        print(price / 12)

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


def plot_regression(df, predicted):
    predicted_prices = pd.DataFrame(predicted, columns=["Predicted Yearly Rent Price"])
    predicted_prices["Year"] = df.index

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
        order=1,
        # lowess=True,
        # robust=True,
        truncate=False,
        # marker='d'
        scatter_kws={"color": "#31A1ED"},
        line_kws={"color": "green"},
    )
    ax.xaxis.set_major_formatter(date_display_from_float)
    ax.tick_params(labelrotation=45)
    ax.set(xlabel="Year", ylabel="Predicted Yearly Rent Price ($)")

    return (ax, predicted_prices)


def plot_reg_resid(predicted_prices, test_data):
    fig, ax = plt.subplots()

    predicted_prices["Residual"] = [
        exp - actual
        for exp, actual in zip(
            predicted_prices["Predicted Yearly Rent Price"], test_data
        )
    ]
    sns.residplot(x="yearfloat", y="Residual", data=predicted_prices)

    ax.xaxis.set_major_formatter(date_display_from_float)
    ax.tick_params(labelrotation=45)
    ax.set(xlabel="Year")

    return ax


@plt.FuncFormatter
def date_display_from_float(datenum, pos):
    return dates.num2date(datenum).strftime("%Y")


if __name__ == "__main__":
    main()
