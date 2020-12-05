# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn import preprocessing
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model,load_model
import keras.backend as K





app = Flask(__name__)

def encode(le,labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)
def decode(le,one_hot):
    dec = np.argmax(one_hot,axis=1)
    return le.inverse_transform(dec)
def ELMoEmbedding(x):
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default" ,as_dict = True)["default"]

def model(message):
    
    data = pd.read_csv("spam.csv",encoding='latin-1')
    y = list(data['v1'])
    x = list(data['v2'])
    le = preprocessing.LabelEncoder()
    le.fit(y)
    x_enc = x
    y_enc = encode(le,y)
    x_train = np.asarray(x_enc[:3850])
    y_train = np.asarray(y_enc[:3850])
    x_test = np.asarray(x_enc[3850:])
    y_test = np.asarray(y_enc[3850:])
    input_text = Input(shape=(1,) ,dtype=tf.string)
    embedding = Lambda(ELMoEmbedding,output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu')(embedding)
    pred = Dense(2, activation='softmax')(dense)
    model = Model(inputs =[input_text],outputs =pred)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    testing_list=[]
    testing_list.append(message)
    testing_list.append('')
    pred_test = np.asarray(testing_list)
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model.load_weights('./elmo_model.h5')
        #predicts = model.predict(x_test,batch_size=32)
        predicts = model.predict(pred_test)
        y_test = decode(le,y_test)
        y_preds = decode(le,predicts)
    return y_preds[0]



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        result = model(message)
        print("hello")

    	
        if result=='ham':
            my_prediction=1
        else:
            my_prediction=0
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)