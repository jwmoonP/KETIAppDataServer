'''
Created on Apr 23, 2019

@author: moonjaewon
'''

import json
import numpy as np
from convertdate import data
import pandas as pd
from flask import Flask, render_template, session, request, g, send_file, redirect, flash
from flask_bootstrap import Bootstrap

import base64
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib


from data_prediction import data_causality as dpc
from data_prediction import data_analysis as da_da
from data_management import data_manager as dm
import VIBES_data

matplotlib.use('Agg')

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)
app.secret_key = "super secret key"

start='2020-09-30 00:00:01'
#end = (dt.date.today()).strftime('%Y-%m-%d %H:%M:%S') 
end ='2020-10-18 00:00:00'

########################################## Server feature
Dset = {}
Dinfo={}
datasetName =""
visual_resample_min='60min'
########################################## Template 
#g.start, g.end
#g.in_feature_list
#g.out_feature_list, g.out_feature_list_add 
###########################################
@app.route('/',  methods=['GET'])
def index():
    # g.start, g.end, g.farm_feature_list, g.out_feature, g.out_feature_list_add
    #global start,end
    ####
    g.start = start
    g.end = end
    g.in_feature_list = ['no_data']
    g.out_feature_list = ['no_data']
    g.out_feature_list_add = ['no data']
    #####
    return render_template('/individual/index.html')

@app.route('/set_date',  methods=['GET'])
def set_date():
    global start, end
    g.start = start = request.args.get('start')
    g.end = end = request.args.get('end')

    if(datasetName!=""):
        try:
           get_data(datasetName)
        except KeyError:
           flash("failed")
           return redirect(request.referrer)
    return redirect(request.referrer)

@app.route('/home/<string:dataset_name>',  methods=['GET'])
def home(dataset_name):
    global datasetName
    datasetName=dataset_name
    
    if(datasetName!=""):
        try:
            get_data(datasetName)
        except KeyError:
            flash("failed")
            return redirect(request.referrer)

    return redirect('/data_visual/'+Dinfo.default_feature)

def get_data(datasetName):
    global Dset
    global Dinfo

    space_list = [datasetName]
    # It depends on domain knowledge.
    ## Insert the code selecting class
    ## Make it clearer and more general.
        
    Dinfo = VIBES_data.VIBE(datasetName)
    Dinfo.set_time(start, end)
    dataD = dm.Data(Dinfo)
    Dset = dm.get_dataSet(space_list, dataD, Dinfo)
    ## Generalization
    ##
    
    return redirect('/data_visual/'+ Dinfo.default_feature)

@app.route('/data_visual/<string:feature>',  methods=['GET'])
def data_visual(feature):

    ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####
    # for visual
    data  = Dset.D_merged[datasetName][feature]
    describe = data.describe().round(2).to_dict()
    data = data.resample(visual_resample_min, label='left').mean()
    data_ = list(data.values)
    category_ = list(data.index.strftime("%Y-%m-%d %H:%M").tolist())
    return render_template('/individual/data_visual.html', data = data_ , category = category_, feature = feature, describe = describe)


@app.route('/data_ingestion_outlier/<string:feature>',  methods=['GET'])
def data_ingestion_outlier(feature):
    
    ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####

    # for visual
    data = Dset.D_original[datasetName][list(Dset.D_original[datasetName].keys())[0]][feature].round(2)
    data_outlier = Dset.D_clean[datasetName][feature].round(2)
    data1_ = list(data.values)
    data2_ = list(data_outlier.values)
    data1_index = list(data.index.strftime("%Y-%m-%d %H:%M").tolist())
    data2_index = list(data_outlier.index.strftime("%Y-%m-%d %H:%M").tolist())

    return render_template('/individual/data_ingestion_outlier.html', data1 = data1_ , data1_index = data1_index, data2_index= data2_index, feature = feature, data2=data2_)

# data correlation
@app.route('/data_correlation',  methods=['GET'])
def data_correlation():
    ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####
    data = Dset.D_clean[datasetName]
    data = data[Dinfo.in_feature_list+Dinfo.out_feature_list].sort_index(axis=1)
    corr = data.corr()
    corr = corr.stack().sort_index(level=1).reset_index()
    corr.columns = ['group', 'variable', 'value']
    corr['value'] = (corr['value'] * 100).astype(int)
    corr = corr.to_json(orient='records')
    return render_template('/individual/data_correlation.html', corr =corr)

# data causaliaty
@app.route('/data_causality_analysis/<string:val1>/<string:val2>',  methods=['GET'])
def data_causality_analysis(val1, val2):
     ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####
    
    data = Dset.D_scaled[datasetName]
    
    feature1_list=Dinfo.in_feature_list + Dinfo.out_feature_list + Dinfo.out_feature_list_add
    feature2_list=Dinfo.in_feature_list
   
    lag_range = 24
    rs, title = dpc.timelag_correlation(data, val1, val2, lag_range)
    max_idx, min_idx, absmax_idx, absmax, min_max_diff, best_lag = dpc.get_specific_rs_information(rs)

    return render_template('/individual/data_causality_analysis.html', val1 = val1, val2= val2, lag_range=lag_range, max_idx = max_idx, min_idx = min_idx, 
                           min_max_diff= min_max_diff, best_lag = best_lag ,absmax_idx = absmax_idx, absmax = absmax , feature1_list = feature1_list, feature2_list = feature2_list)

@app.route('/fig_causality/<string:val1>/<string:val2>/<int:lag_range>',  methods=['GET'])
def fig_causality(val1, val2, lag_range):
 
    data = Dset.D_scaled[datasetName]
    rs, title = dpc.timelag_correlation(data, val1, val2, lag_range)
    img = dpc.timelag_correlation_fig(rs, lag_range, val1, val2, title)

    return send_file(img, mimetype='image/png')

@app.route('/data_causality_result',  methods=['GET'])
def data_causality_result():
    ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####
    
    data = Dset.D_scaled[datasetName]
    
    print("What is data_causality_result?")
    feature1_list=Dinfo.in_feature_list + Dinfo.out_feature_list + Dinfo.out_feature_list_add
    feature2_list=Dinfo.in_feature_list
    print(data.head())

    lag_range = 24
    
    farm_causal_history =  dpc.get_causal_history_table(data, feature1_list, feature2_list, lag_range)
    high_causal_table = dpc.get_high_causal_table(farm_causal_history, feature2_list)
    high_causal_json = dpc.get_high_causal_dict(high_causal_table)
    print(high_causal_json)
    return render_template('/individual/data_causality_result.html', high_causal_json = high_causal_json)
    

@app.route('/data_distribution/<string:feature>',  methods=['GET'])
def data_distribution(feature):
    ####
    g.start = start
    g.end = end
    g.in_feature_list = Dinfo.in_feature_list
    g.out_feature_list = Dinfo.out_feature_list
    g.out_feature_list_add = Dinfo.out_feature_list_add
    #####
    data_list = Dinfo.in_feature_list+Dinfo.out_feature_list+Dinfo.out_feature_list_add
    return render_template('/individual/data_distribution.html', data_list = data_list, feature = feature)

@app.route('/fig_data_distribution/<string:feature>',  methods=['GET'])
def fig_data_distribution(feature):
    
    data = Dset.D_clean[datasetName]
    img = da_da.plot_data_distribution(data, feature)
    return send_file(img, mimetype='image/png')

@app.route('/fig_acf_pacf/<string:feature>',  methods=['GET'])
def fig_acf_pacf(feature):
    data = Dset.D_clean[datasetName]
    img = da_da.plot_acf_pacf(data, feature)
    return send_file(img, mimetype='image/png')
"""
@app.route('/farm_status_correlation/<string:feature>/<int:maxlag>/<string:datatype>',  methods=['GET'])
def farm_status_correlation(feature, maxlag, datatype):
    global start,end
    g.start = start
    g.end = end
    global farm_feature_list
    g.farm_feature_list = farm_feature_list
    g.out_feature_list = out_feature_list
    g.out_feature_list_add = out_feature_list_add
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
    return render_template('/individual/farm_status_correlation.html', maxlag = maxlag, feature = feature, datatype = datatype, data_list = list(data.columns), datatype_list= datatype_list)


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
"""
####################

###################### Server 2

if __name__ == '__main__':
    app.run(debug=True, port='9008')


