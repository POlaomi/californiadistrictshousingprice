import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names,a,b):
        self.attribute_names = attribute_names
        self.a = a
        self.b = b
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values
