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


def process(df):
    pipeline = Pipeline(
        memory=None,
        steps=[
            ("standardize", StandardScaler()),
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
    process(DataReader.create_combined_df())
