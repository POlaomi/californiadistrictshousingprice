#import the required libraries
import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import seaborn as sns
from dataframe_selector import *
from categorical_encoder import *
from attributes_adder import *
#from django.shortcuts import render

#from sklearn.base import BaseEstimator, TransformerMixin 

    

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








