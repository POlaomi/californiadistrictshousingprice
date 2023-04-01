
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names,a,b):
        self.attribute_names = attribute_names
        self.a = a
        self.b = b
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values
  
"""
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names,a,b):
        self.attribute_names = attribute_names
        self.a = a
        self.b = b
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        return X.iloc[:,self.a:self.b].values
    """
    
