from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 
import yfinance as yf 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io 
import base64
from model import GridSearchARIMA, convert_timeseries_stationary
from joblib import load

def get_stock_data(stock_name):
    return yf.download(stock_name,
                       start='2015-01-01',
                       end='2021-01-01') 

def requestResults(stock_name):
    df = get_stock_data(stock_name)
    stat_data = convert_timeseries_stationary(df['Close'])
    d, ts = stat_data.values()
    model = GridSearchARIMA(ts,d)
    curr_time = ts.index.values[-1]
    predictions = model.predict(curr_time,curr_time + np.timedelta64(30*4,'D'))
    return pd.DataFrame(predictions)


def _create_plot(data,title):
    """ Create a matplotlib plot without the need to locally save it.
        Source: https://gitlab.com/-/snippets/1924163
    """ 
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(title)
    axis.grid()
    axis.plot(data)

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    return pngImageB64String


app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/',methods=['POST','GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success',name=user))

@app.route('/success/<name>')
def success(name):
    stock = get_stock_data(name)
    img1 = _create_plot(stock['Close'],'Stock price')

    return render_template("analysis.html", name=name, data=requestResults(name).to_html(), image=img1) 


if __name__ == '__main__':
    app.run(debug=True)
