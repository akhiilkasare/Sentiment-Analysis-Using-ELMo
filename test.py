from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import load_model, load_weights

app = Flask(__name__)

MODEL_PATH = '/home/akhil/Downloads/deep_learning/elmo/elmo_model.h5'
model = load_weights(MODEL_PATH)
    
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = model.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
    