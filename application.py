import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pf
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
logistic_model=pickle.load(open('ML project/models/logistic.pkl','rb'))
standard_scaler=pickle.load(open('ML project/models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        Humidity= float(request.form.get('Humidity'))
        PM2=float(request.form.get('PM2'))
        PM10=float(request.form.get('PM10'))
        NO2=float(request.form.get('NO2'))
        SO2=float(request.form.get('SO2'))
        CO=float(request.form.get('CO'))
        Proximity_to_Industrial_Areas=float(request.form.get('Proximity_to_Industrial_Areas'))
        Population_Density=float(request.form.get('Population_Density'))
        
        

        new_data_scaled=standard_scaler.transform([[Temperature,Humidity,PM2,PM10,NO2,SO2,CO,Proximity_to_Industrial_Areas,Population_Density]])
        result=logistic_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
