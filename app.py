from flask import Flask, render_template, request, redirect, url_for
import numpy as np 
import pandas as pd 

import matplotlib 
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import seaborn as sns 
import yfinance as yf 

import os
import io 
import base64
from model import GridSearchARIMA, convert_timeseries_stationary

matplotlib.use('Agg')  # turn off gui
sns.set(rc={'figure.figsize':(8,3)}) # Use seaborn backend 

def get_stock_data(stock_name):
    return yf.download(stock_name,
                       start='2015-01-01',
                       end='2021-11-07') 

def requestResults(stock_name):
    df = get_stock_data(stock_name)
    stat_data = convert_timeseries_stationary(df['Close'])
    d, ts = stat_data.values()
    model = GridSearchARIMA(ts,d)
    curr_time = ts.index.values[-1]
    predictions = model.predict(curr_time,curr_time + np.timedelta64(30*10,'D'))
    return pd.DataFrame(predictions)

def extract_plot_from_fig(fig):
    """ Create a matplotlib plot without the need to locally save it.
        Source: https://gitlab.com/-/snippets/1924163
    """ 
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    return pngImageB64String


PLOTS_FOLDER = os.path.join('static','plots')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PLOTS_FOLDER
 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/',methods=['POST','GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success',name=user))

@app.route('/stock/<name>')
def success(name):
    df = get_stock_data(name)

    volume_fig,ax = plt.subplots()
    sns.lineplot(x=df.index,y=df['Volume'],label='Volume',ax=ax)
    sns.lineplot(x=df.index,y=df['Volume'].rolling(window=12).mean(),label='Averaged volume',ax=ax)
    ax.set_title('Volume of stock over time')
    volume_img = extract_plot_from_fig(volume_fig)
    

    fig, (ax1,ax2,ax3) = plt.subplots(3,figsize=(8,5))
    df.groupby(df.index.day).mean().plot(y=['High','Close','Low'],ax=ax1,xlabel='Day');
    df.groupby(df.index.month).mean().plot(y=['High','Close','Low'],ax=ax2, xlabel='Month');
    df.groupby(df.index.year).mean().plot(y=['High','Close','Low'], ax=ax3, xlabel='Year');
    plt.tight_layout()
    stock_img = extract_plot_from_fig(fig)

    decompose_fig = seasonal_decompose(df['Close'],period=365,model='additive').plot();
    decompose_fig.set_size_inches(10,5)
    decompose_img = extract_plot_from_fig(decompose_fig)

    acf_fig = plot_acf(df['Close'].dropna());
    pacf_fig = plot_pacf(df['Close'].diff(2).dropna());

    acf_img = extract_plot_from_fig(acf_fig);
    pacf_img = extract_plot_from_fig(pacf_fig);

    predictions = requestResults(name)
    
    pred_fig = plt.figure()
    df['Close'].plot(fig=pred_fig);
    predictions['predicted_mean'].plot(fig=pred_fig);
    pred_img = extract_plot_from_fig(pred_fig)



    return render_template("analysis.html", 
                            name=name, 
                            data=pd.DataFrame(df.describe()['Close']).to_html(), 
                            image=volume_img,
                            image1=stock_img,
                            image2=decompose_img,
                            image3=acf_img,
                            image4=pacf_img,
                            image5=pred_img)


if __name__ == '__main__':
    app.run(debug=True)
