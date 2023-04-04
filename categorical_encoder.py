import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ohe = OneHotEncoder()):
        self.ohe = ohe
    def fit(self, X,y=None,):
        return self.ohe.fit(X)
    def transform(self, X, y=None):
        return self.ohe.transform(X).toarray()