from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame


class Applier(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)
        return X.select_dtypes(include=[self.dtype])
