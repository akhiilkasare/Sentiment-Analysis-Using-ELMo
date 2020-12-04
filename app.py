 #Importing essential libraries
from flask import Flask, render_template, request
from sklearn.externals import joblib
import pickle
import numpy as np

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = '/home/akhil/Downloads/deep_learning/elmo/elmo_model.h5'
#classifier = pickle.load(open(filename, 'rb'))
classifier = joblib.load('test_elmo.pkl')
cv = pickle.load(open('/home/akhil/Downloads/deep_learning/elmo/elmo_model.h5','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(data)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)


