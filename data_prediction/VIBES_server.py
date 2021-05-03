'''
Created on Apr 23, 2019

@author: moonjaewon
'''
import os
import sys
import json

import numpy as np
from convertdate import data
import pandas as pd
sys.path.append("../..")
from flask import Flask , render_template, session, request, g, send_file
from flask_bootstrap import Bootstrap
import base64

import data_manager.data_manager as dm
import data_manager.data_outlier as do

import matplotlib.pyplot as plt
import data_prediction.data_causality as dpc
import data_manager.data_preprocessing as pre

import data_prediction.data_analysis as da_da
import matplotlib
from flask import g
matplotlib.use('Agg')

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)
app.secret_key = "super secret key"

VIBE_data={}
VIBE_pre={}

VIBE_scaled =pd.DataFrame()
VIBE = pd.DataFrame()
########################################## Server 1
farm_feature_list = ['CO2ppm', 'H2Sppm', 'NH3ppm', 'OptimalTemperature', 'RichHumidity', 'RichTemperature']
out_feature_list =['out_humid','out_sunshine', 'out_temp', 'out_wind_speed']
out_feature_list_add=['out_O3', 'out_PM10','out_rainfall']

#######
@app.route('/home/<string:dbname>',  methods=['GET'])
def home(dbname): #KDS1, KDS2
    
    global VIBE_data, VIBE_pre
    global scaler, VIBE_scaled, VIBE
    global farm_feature_list
    if dbname == 'KDS1':
        farm_feature_list = ['CO2ppm', 'H2Sppm', 'NH3ppm']
        start='2020-07-11 23:59:59'#洹� �떎�쓬�궇遺��꽣
        end='2020-07-25 00:00:00'
        sample_start ='2020-07-11'
        sample_end ='2020-07-15'
        
    elif dbname =='KDS2':
        farm_feature_list = ['CO2ppm', 'H2Sppm', 'NH3ppm', 'RichHumidity', 'RichTemperature']
        start='2020-07-11 23:59:59'#洹� �떎�쓬�궇遺��꽣
        end='2020-07-25 00:00:00'
        sample_start ='2020-07-11'
        sample_end ='2020-07-15'
        
    g.farm_feature_list = farm_feature_list
    
    """ Server 1
    start='2020-04-22 23:59:59'#洹� �떎�쓬�궇遺��꽣
    end='2020-05-06 00:00:00'
    sample_start ='2020-04-22'
    sample_end ='2020-04-23'
    VIBE_data['out_weather'], VIBE_pre['out_weather'], VIBE_data['out_air'], VIBE_pre['out_air'], VIBE_data['farm_air'], VIBE_pre['farm_air'], VIBE_data['farm_weather'], VIBE_pre['farm_weather'] = dm.get_influx_data(id_, start, end, sample_start, sample_end)
    integration_list =[VIBE_pre['farm_air'], VIBE_pre['farm_weather'], VIBE_data['out_air'], VIBE_data['out_weather']]
    """
    #Server 2
    VIBE_data['farm'], VIBE_pre['farm'] = dm.get_processed_data(dbname, start, end, sample_start, sample_end)
    dbname='Outdoor_air_VIBES'
    VIBE_data['out_air'], VIBE_pre['out_air']  = dm.get_processed_data(dbname, start, end, sample_start, sample_end)
    dbname='Outdoor_weather_Sangju'
    VIBE_data['out_weather'], VIBE_pre['out_weather'] = dm.get_processed_data(dbname, start, end, sample_start, sample_end)

    integration_list =[VIBE_data['farm'], VIBE_data['out_air'], VIBE_data['out_weather']]
    integration_min ='1H'
    VIBE = dm.data_integration(integration_list, integration_min)
    DP = pre.data_transformation()
    scaler, VIBE_scaled = DP.data_scaling(VIBE)
    return render_template('home.html')

@app.route('/data_visual/<string:dataname>/<string:feature>',  methods=['GET'])
def data_visual(dataname, feature):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE_data, VIBE_pre
    global scaler, VIBE_scaled

    # for visual
    describe = VIBE_data[dataname][feature].describe().round(2).to_dict()
    print(describe)
    print(VIBE_data[dataname]['2020-07-12 13:30:00':'2020-07-12 15:30:00'])
    data = VIBE_data[dataname].resample('30min', label='left').min()
    print(data[:60])
    data_ = list(data[feature].values)
    
    category_ = list(data[feature].index.strftime("%Y-%m-%d %H:%M").tolist())
    return render_template('data_visual.html', data = data_ , category = category_, feature = feature, describe = describe)


@app.route('/data_ingestion_outlier/<string:dataname>/<string:feature>',  methods=['GET'])
def data_ingestion_outlier(dataname, feature):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE_data, VIBE_pre
    global scaler, VIBE_scaled

    # for visual
    data = VIBE_data[dataname].round(2)
    data_outlier = VIBE_pre[dataname].round(2)
    data1_ = list(data[feature].values)
    data2_ = list(data_outlier[feature].values)
    category_ = list(data[feature].index.strftime("%Y-%m-%d %H:%M").tolist())

    return render_template('data_ingestion_outlier.html', data1 = data1_ , category = category_, feature = feature, data2=data2_)

# data correlation
@app.route('/data_correlation',  methods=['GET'])
def data_correlation():
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE
    data = VIBE
    data = data[farm_feature_list+out_feature_list].sort_index(axis=1)
    corr = data.corr()
    corr = corr.stack().sort_index(level=1).reset_index()
    corr.columns = ['group', 'variable', 'value']
    corr['value'] = (corr['value'] * 100).astype(int)
    corr = corr.to_json(orient='records')
    
    return render_template('/data_correlation.html', corr =corr)

# data causaliaty
@app.route('/data_causality_analysis/<string:val1>/<string:val2>',  methods=['GET'])
def data_causality_analysis(val1, val2):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE_scaled
    data = VIBE_scaled
    feature1_list=out_feature_list + out_feature_list_add + farm_feature_list
    feature2_list=farm_feature_list
    
    lag_range = 24
    rs, title = dpc.timelag_correlation(data, val1, val2, lag_range)
    max_idx, min_idx, absmax_idx, absmax, min_max_diff, best_lag = dpc.get_specific_rs_information(rs)

    return render_template('data_causality_analysis.html', val1 = val1, val2= val2, lag_range=lag_range, max_idx = max_idx, min_idx = min_idx, 
                           min_max_diff= min_max_diff, best_lag = best_lag ,absmax_idx = absmax_idx, absmax = absmax , feature1_list = feature1_list, feature2_list = feature2_list)

@app.route('/fig_causality/<string:val1>/<string:val2>/<int:lag_range>',  methods=['GET'])
def fig_causality(val1, val2, lag_range):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE_scaled
    data = VIBE_scaled
    rs, title = dpc.timelag_correlation(data, val1, val2, lag_range)
    img = dpc.timelag_correlation_fig(rs, lag_range, val1, val2, title)

    return send_file(img, mimetype='image/png')

@app.route('/data_causality_result',  methods=['GET'])
def data_causality_result():
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    
    global VIBE_scaled
    data = VIBE_scaled
    print("What is data_causality_result?")
    feature1_list=out_feature_list + out_feature_list_add + farm_feature_list
    feature2_list=farm_feature_list
    print(data.head())
    print(VIBE_scaled.columns)
    lag_range = 24
    
    farm_causal_history =  dpc.get_causal_history_table(data, feature1_list, feature2_list, lag_range)
    high_causal_table = dpc.get_high_causal_table(farm_causal_history, feature2_list)
    high_causal_json = dpc.get_high_causal_dict(high_causal_table)
    print(high_causal_json)
    return render_template('data_causality_result.html', high_causal_json = high_causal_json)
    

@app.route('/data_distribution/<string:feature>',  methods=['GET'])
def data_distribution(feature):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    data = VIBE
    data_list = farm_feature_list+out_feature_list+out_feature_list_add
    return render_template('/data_distribution.html', data_list = data_list, feature = feature)

@app.route('/fig_data_distribution/<string:feature>',  methods=['GET'])
def fig_data_distribution(feature):
    
    global VIBE
    data = VIBE
    img = da_da.plot_data_distribution(data, feature)
    return send_file(img, mimetype='image/png')

@app.route('/fig_acf_pacf/<string:feature>',  methods=['GET'])
def fig_acf_pacf(feature):
    global VIBE
    data = VIBE
    img = da_da.plot_acf_pacf(data, feature)
    return send_file(img, mimetype='image/png')

@app.route('/farm_status_correlation/<string:feature>/<int:maxlag>/<string:datatype>',  methods=['GET'])
def farm_status_correlation(feature, maxlag, datatype):
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    global VIBE, VIBE_scaled
    datatype_list = ['original', 'original_scale', 'original_diff', 'original_diff_scale']
    
    if datatype ==datatype_list[0]:
        data = VIBE[farm_feature_list]
    if datatype ==datatype_list[1]:
        data = VIBE_scaled[farm_feature_list]
    elif datatype ==datatype_list[2]:
        data = VIBE[farm_feature_list].diff().dropna()
    elif datatype ==datatype_list[3]:
        scaler, data = DP.data_scaling(VIBE[farm_feature_list].diff().dropna())
    else:
        data = VIBE[farm_feature_list]
    
    table = dc.grangers_causation_matrix(maxlag, data, variables= data.columns, test='ssr_chi2test')
    return render_template('/farm_status_correlation.html', maxlag = maxlag, feature = feature, datatype = datatype, data_list = list(data.columns), datatype_list= datatype_list)


@app.route('/fig_grangers_causation/<string:feature>/<int:maxlag>/<string:datatype>',  methods=['GET'])
def fig_grangers_causation(feature,maxlag, datatype):
    
    global VIBE, VIBE_scaled
    datatype_list = ['original', 'original_scale', 'original_diff', 'original_diff_scale']
    
    if datatype ==datatype_list[0]:
        data = VIBE[farm_feature_list]
    if datatype ==datatype_list[1]:
        data = VIBE_scaled[farm_feature_list]
    elif datatype ==datatype_list[2]:
        data = VIBE[farm_feature_list].diff().dropna()
    elif datatype ==datatype_list[3]:
        scaler, data = DP.data_scaling(VIBE[farm_feature_list].diff().dropna())
    else:
        data = VIBE[farm_feature_list]
        
    table = dc.grangers_causation_matrix(maxlag, data, variables= data.columns, test='ssr_chi2test')
    img = dc.plot_causal_table(table, feature+'_y')
    return send_file(img, mimetype='image/png')

########################################## Server 2

if __name__ == '__main__':
    app.run(debug=True)

