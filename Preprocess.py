import pandas as pd
import numpy as np
from sklearn import feature_selection as ftselect
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

# DecisionTreeRegressor
import DataReader
from Applier import Applier

KFOLDS = 10


def process(df):
    pipeline = Pipeline(
        memory=None,
        steps=[
            ("standardize", StandardScaler()),
            ("feat_select", ftselect.SelectFromModel(RandomForestRegressor())),
            (
                "regressor",
                ElasticNetCV(cv=len(df.index), max_iter=1000, normalize=False),
            ),  # AdaBoostRegressor()
        ],
    )

    return pipeline


if __name__ == "__main__":
    process(DataReader.create_combined_df())
