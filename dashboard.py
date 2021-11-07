from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 
import yfinance as yf 

from model import GridSearchARIMA, convert_timeseries_stationary
from joblib import load

def get_stock_data(stock_name):
    return yf.download(stock_name,
                       start='2015-01-01',
                       end='2021-01-01')

def requestResults(stock_name):
    stock = get_stock_data(stock_name)
    stat_data = convert_timeseries_stationary(stock['Close'])
    ts, d = stat_data.values()
    model = GridSearchARIMA(ts,d)
    curr_time = ts.index.values[-1]
    predictions = model.predict(curr_time,curr_time + np.timedelta64(30*4,'D'))
    return str(predictions) + '\n\n'
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('templates/home.html')

@app.route('/',methods=['POST','GET'])
def get_data():
    if requests.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success',name=user))

@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + "</xmp>"


if __name__ == '__main__':
    app.run(debug=True)