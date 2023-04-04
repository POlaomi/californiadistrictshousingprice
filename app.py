#import the required libraries
import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
#from django.shortcuts import render

from sklearn.base import BaseEstimator, TransformerMixin 
    
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
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ohe = OneHotEncoder()):
        self.ohe = ohe
    def fit(self, X,y=None,):
        return self.ohe.fit(X)
    def transform(self, X, y=None):
        return self.ohe.transform(X).toarray()
    
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names,a,b):
        self.attribute_names = attribute_names
        self.a = a
        self.b = b
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

import __main__
__main__.DataFrameSelector = DataFrameSelector
__main__.CategoricalEncoder = CategoricalEncoder
__main__.AttributesAdder = AttributesAdder


#create an app
app = Flask(__name__)
#load the model
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
#load the preprocessing pipeline
pipeline = pickle.load(open('preprocessing_pp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(pd.DataFrame(np.array(list(data.values())).reshape(1,-1), columns=list(data.keys())))
    new_data = pipeline.transform(pd.DataFrame(np.array(list(data.values())).reshape(1,-1), columns=list(data.keys())))
    output = rf_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    column_names = [x for x in request.form.keys()]
    processed_input = pipeline.transform(pd.DataFrame(np.array(list(data)).reshape(1,-1), columns=column_names))
    print(processed_input)
    output = rf_model.predict(processed_input)[0]
    return render_template("home.html", prediction_text="The District House prediction Price is {}".format(output))
    



if __name__ == "__main__":
    app.run(debug = True)







