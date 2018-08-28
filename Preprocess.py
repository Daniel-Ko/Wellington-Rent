import pandas as pd
import numpy as np
from sklearn import feature_selection as ftselect, cross_validation as cvalid
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV
from sklearn.grid_search import GridSearchCV

# DecisionTreeRegressor
import DataReader
from Applier import Applier

KFOLDS = 10


def process():
    df = DataReader.create_combined_df()

    pipeline = Pipeline(
        memory=None,
        steps=[
            (
                "feat_union",
                FeatureUnion(
                    n_jobs=1,
                    transformer_list=[
                        (
                            "normalised",
                            Pipeline([("scaler", StandardScaler())]),
                        )  # ("apply_df", Applier(np.number))
                    ],
                ),
            ),
            (
                "feat_select",
                ftselect.SelectFromModel(RandomForestRegressor())
                # ftselect.SelectKBest(k=5),
            ),
            (
                "regressor",
                ElasticNetCV(cv=len(df.index), max_iter=1000, normalize=False),
            ),  # AdaBoostRegressor()
        ],
    )
    predicted = pipeline.fit(df, df["Wellington"]).predict(df)
    score = pipeline.score(df, df["Wellington"])
    print(f"SCORE: {score}")

    feat_importances = pipeline.named_steps[
        "feat_select"
    ].estimator_.feature_importances_

    feat_support = pipeline.named_steps["feat_select"].get_support()

    sig_feats = pd.DataFrame(
        feat_support.reshape(-1, len(feat_support)),
        index=["Important feature?"],
        columns=df.columns,
    )
    # for i, score in enumerate(feat_importances):
    sig_feats.loc["Importance score"] = feat_importances
    print(sig_feats)
    # display(sig_feats)

    # scores = cvalid.cross_val_score(
    #     pipeline,
    #     df,
    #     df["Wellington"],
    #     verbose=1,
    #     scoring="r2",
    #     # cv=KFold(n_splits=KFOLDS),
    # )

    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return pipeline


if __name__ == "__main__":
    process()
