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
    length = request.form.get('length')
    diameter = request.form.get('diameter')
    height = request.form.get('height')
    # LoanDuration = request.form.get('duration')              
   
    ##Test model prediction with static data. Reshape to change to 2D array 
#     testdata = np.reshape([
#     None,
#     length,
#     diameter,
#     height,
#     None,
#     None,
#     None,
#     None
#     ],(1, -1))
    
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

#     txt = "The predicted abalone age is {}.".format(pred_result[0])
    
#     predictions = {"txt":txt}

    # if(pred_result[0]==0):
    #     txt = 'No Risk Loan'
    # else:
    #     txt = 'Risky Loan'
    # print(txt)
    
    return render_template('index.html', prediction_text='The predicted abalone age is: {}'.format(pred_result))
#     return render_template('index.html', prediction_text=txt)

if __name__ == "__main__":
    app.run()
