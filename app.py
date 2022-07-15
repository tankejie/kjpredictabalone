from flask import Flask, request, jsonify, url_for, redirect, render_template
import numpy as np
import pickle
import requests

app = Flask(__name__)
model = pickle.load(open('model.h5','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    # Age = request.form.get('Rings')
    # if(Gender=='Female'):
    #     Gender=0
    # else:
    #     Gender=1
    # MStatus = request.form.get('mstatus')
    # if(MStatus=='Single'):
    #     MStatus=0
    # else:
    #     MStatus=1
    length = float(request.form.get('length'))
    diameter = float(request.form.get('diameter'))
    height = float(request.form.get('height'))
    # LoanDuration = request.form.get('duration')              
   
    ##Test model prediction with static data. Reshape to change to 2D array 
    testdata = np.reshape([
    0,
    length,
    diameter,
    height,
    0,
    0,
    0,
    0
    ],(1, -1))

    pred_result = model.predict(testdata)

    

    # if(pred_result[0]==0):
    #     txt = 'No Risk Loan'
    # else:
    #     txt = 'Risky Loan'
    # print(txt)
#     return render_template('index.html', prediction_text=prediction_text)
    return render_template('index.html', prediction_text='The predicted abalone age is: {:.2f}.'.format(pred_result))
    # return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
