import sklearn
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
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ohe = OneHotEncoder()):
        self.ohe = ohe
    def fit(self, X,y=None,):
        return self.ohe.fit(X)
    def transform(self, X, y=None):
        return self.ohe.transform(X).toarray()
    
class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, 3] / X[:,6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,4]/ X[:,3]
            return np.c_[X,rooms_per_household, bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household]
  
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
    
