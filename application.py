import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app=application

# Load Ridge regression model and StandardScaler
ridge_model= pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))  # use correct scaler filename

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
         
         Temperature = float(request.form.get('Temperature'))
         RH = float(request.form.get('RH'))
         Ws = float(request.form.get('Ws'))
         Rain = float(request.form.get('Rain'))
         FFMC = float(request.form.get('FFMC'))
         DMC = float(request.form.get('DMC'))
         ISI = float(request.form.get('ISI'))
         Classes = float(request.form.get('Classes'))
         Region = float(request.form.get('Region'))

        
         new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
         result=ridge_model.predict(new_data_scaled)

         return render_template('home.html',results=result[0])



          
    else:
        return render_template('home.html')
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
