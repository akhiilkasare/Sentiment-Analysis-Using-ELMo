#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:26:57 2020

@author: akhil
"""

from flask import Flask, render_template, request,url_for

import numpy as np

import pandas as pd
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import load_model



app = Flask(__name__)

def model():
    model = load_model('/home/akhil/Downloads/deep_learning/elmo/elmo_model.h5')
    
    
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST', "GET"])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = model.transform(data)
    	my_prediction = classifier.predict(data)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)


