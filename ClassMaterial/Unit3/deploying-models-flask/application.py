import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/SMSSpamCollection.txt', sep='\t', header=None)
df.columns = ['target', 'msg']
y = df['target']
X = df['msg']

# Tfidf, filter stop words, 300 features
cvec = TfidfVectorizer(stop_words='english', max_features = 300)
X = cvec.fit_transform(X)
clf = MultinomialNB()
clf.fit(X, y)

import pickle

with open('titanic_tree.pkl', 'rb') as picklefile:
    PREDICTOR = pickle.load(picklefile)

#-------- ROUTES GO HERE -----------#
@app.route('/is_spam', methods=["GET"])
def is_spam():
    msg = pd.Series(flask.request.args['msg'])
    X_new = cvec.transform(msg)
    score = clf.predict(X_new)
    results = {'prediction': score[0]}
    return flask.jsonify(results)

@app.route('/predict', methods=["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = [pclass, sex, age, fare, sibsp]
    score = PREDICTOR.predict_proba(np.array([item]))
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)

@app.route('/page')
def page():
   with open("page.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       pclass = inputs['pclass'][0]
       sex = inputs['sex'][0]
       age = inputs['age'][0]
       fare = inputs['fare'][0]
       sibsp = inputs['sibsp'][0]

       item = np.array([pclass, sex, age, fare, sibsp])
       score = PREDICTOR.predict_proba(item.reshape(1, 5))
       results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       return flask.jsonify(results)

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(debug = True, HOST, PORT)