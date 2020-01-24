# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:00:12 2020

@author: Sriharsha Komera
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model_Fuel = pickle.load(open("model_Fuel.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('CO2 Emission.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_Fuel.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('CO2 Emission.html', prediction_text='CO2 Emission is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model_Fuel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

#if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=8080)

if __name__ == "__main__":
    app.run(debug=True)