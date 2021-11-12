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

from model import ARIMAModel, convert_timeseries_stationary


matplotlib.use('Agg')  # turn off gui
sns.set(rc={'figure.figsize':(8,3)}) # Use seaborn backend 



def get_stock_data(stock_name,start,end):
    return yf.download(stock_name,
                            start=start,
                            end=end) 

def requestResults(df,p,q):
    stat_data = convert_timeseries_stationary(df['Close'])
    d, ts = stat_data.values()
    model = ARIMAModel(ts,p,d,q)
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
        print(request.path)
        user = request.form['search']
        start = request.form['start']
        end = request.form['end']
        return redirect(url_for('success',name=user,start=start,end=end))

@app.route('/stock/<name>/<start>/<end>', methods=['POST','GET'])
def success(name, start,end):
    df = get_stock_data(name,start,end)

    p = None
    q = None 
    train_img = None
    stat_img = None
    predictions=None

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

    
    stat_data = convert_timeseries_stationary(df['Close'])
    d, ts, fig = stat_data.values()
    stat_img = extract_plot_from_fig(fig)

    acf_fig = plot_acf(ts);
    pacf_fig = plot_pacf(ts);

    acf_img = extract_plot_from_fig(acf_fig);
    pacf_img = extract_plot_from_fig(pacf_fig);

    if request.method == 'POST':
        p = int(request.form['p'])
        q = int(request.form['q'])

        model, (fig,ax) = ARIMAModel(ts,p,d,q)

        train_img = extract_plot_from_fig(fig)

        curr_time = ts.index.values[-1]
        predictions = pd.DataFrame(model.predict(curr_time,curr_time + np.timedelta64(30*10,'D'))).to_html()

        

    return render_template("analysis.html", 
                            name=name, 
                            start=start,
                            end=end,
                            stats=pd.DataFrame(df.describe()['Close']).to_html(), 
                            image=volume_img,
                            image1=stock_img,
                            image2=decompose_img,
                            image3=acf_img,
                            image4=pacf_img,
                            image5=stat_img,
                            image6=train_img,
                            p=p,
                            q=q,
                            predictions=predictions)
                           


if __name__ == '__main__':
    app.run(debug=True)
