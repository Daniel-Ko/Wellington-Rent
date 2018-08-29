import pandas as pd
import numpy as np
from sklearn import feature_selection as ftselect
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV

import DataReader


def process(df):
    pipeline = Pipeline(
        memory=None,
        steps=[
            ("standardize", StandardScaler()),
            ("feat_select", ftselect.SelectFromModel(RandomForestRegressor())),
            (
                "regressor",
                # ElasticNetCV(cv=len(df.index), max_iter=1000, normalize=False),
                LassoCV(cv=len(df.index), max_iter=3000, normalize=False),
            ),
        ],
    )

    return pipeline


if __name__ == "__main__":
    process(DataReader.create_combined_df(False, False))
