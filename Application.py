from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
# from xgboost import XGBRegressor

import pickle
import pandas as pd
import numpy as np
import os


# global laptop
app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('df.pkl','rb'))
pred = pickle.load(open('pipe.pkl','rb'))
laptop = pd.read_csv('Cleaned_laptop_data.csv')


@app.route('/')
def index():
    return render_template('index.html')

# for visualization
@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/laptop_price_predict', methods=['GET' , 'POST'])
def laptop_price_predict():
    global laptop

    companies = sorted(laptop['Company'].unique())
    type = sorted(laptop['TypeName'].unique())
    inch = sorted(laptop['Inches'].unique())
    ram=sorted(laptop['Ram'].unique())
    screen=laptop['Touchscreen'].unique()
    ips=laptop['Ips'].unique()
    ppi=laptop['ppi'].unique()
    CPU=laptop['Cpu brand'].unique()
    HDD=sorted(laptop['HDD'].unique())
    SSD=sorted(laptop['SSD'].unique())
    GPU=laptop['Gpu brand'].unique()
    OS=laptop['os'].unique()
    return render_template('laptop_price_predict.html',companies=companies, type=type, inch=inch, ram=ram, screen=screen, ips=ips, ppi=ppi, CPU=CPU, HDD=HDD, SSD=SSD, GPU=GPU, OS=OS)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')
    type=request.form.get('type')
    inch =request.form.get('inch')
    Ram = request.form.get('ram')
    touchscreen = request.form.get('touch')
    ips = request.form.get('ips')
    src = request.form.get('ppi')
    processor=request.form.get('processor')
    HDD=request.form.get('HDD')
    SSD=request.form.get('SSD')
    gpu = request.form.get('gpu')
    os=request.form.get('os')

    inch = float(inch)
    X_res = int(src.split('x')[0])
    Y_res = int(src.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / inch
    # print(ppi)


    prediction = pred.predict(pd.DataFrame(columns=['Company', 'TypeName', 'Inches', 'Ram', 'Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os'],
                                            data=np.array([company,type,inch,Ram,touchscreen,ips,ppi,processor,HDD,SSD,gpu,os]).reshape(1, 12)))
    # print(prediction)
    return str(np.round(prediction[0],2))


if __name__=='__main__':
    app.run(debug=True)
