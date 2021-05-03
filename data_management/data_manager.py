import pandas as pd
import numpy as np
from influxdb import DataFrameClient, InfluxDBClient
from datetime import datetime
import math

import data_management.data_preprocessing as pre
import data_management.feature_engineering as fe
        
class Data:
    def __init__(self,data_info): 
        self.D_original= {};   self.D_sample={}; 
        self.D_merged={}; self.D_clean={}
        self.D_extended={}; self.D_selected={}
        self.D_scaled={};   self.D_scaler={}
        pass
        
    def get_mergedD(self, data, resample_min, drop_features):      
        preD = pre.ProcessedData()
        merged_data = preD.data_integration(data, resample_min)
        valid_data = preD.data_validation(merged_data)
        selected_data = valid_data.drop(drop_features, axis=1, errors='ignore')
        return selected_data

    def get_cleanD(self, data_original, data_sample, data_range, resample_min, limit_minute_for_nan_processing):
                        #original-> VIBE_cl
        mD = pre.ProcessedData()
        data=mD.data_cleaning(data_original, data_sample, data_range)
        merged_data = mD.data_integration(data, resample_min)
        nonnan_data = mD.data_nan_processing(merged_data, resample_min, limit_minute_for_nan_processing)
        
        valid_data = mD.data_validation(merged_data)
        return valid_data
    
    def get_extendedD(self, data, data_range_level, vector_feature_list):   
        #ratio_feature ={'Humidity':'out_humid', 'Temperature':'out_temp'}
        fe_e = fe.feature_engineering()
        #Data_Ex = fe_e.add_level_features(data, data_range_level)
        Data_Ex = fe_e.add_vector_features(data, vector_feature_list)
        Data_Ex = fe_e.add_time_Sine_Cosine_features(data)
        #Data_Ex = fe_e.workday_feature_genaration(data, 'SouthKorea')
        print(Data_Ex.columns)
        return Data_Ex
    
    def get_selectedD(self, Data):   
        """
        Describe code
        """
        Data_sl = Data
        return Data_sl
    
    def get_scaledD(self, Data):
        mD = pre.ProcessedData()
        scaler, data_scale = mD.data_scaling(Data, )
        return scaler, data_scale
    
def get_dataSet(space_list, dataSet, data_info):
    
    for space in space_list:
        # Original and sample Data ingestion from influx DB 
        dataSet.D_original[space] = data_info.data_ingestion(data_info.ingestion_start, data_info.ingestion_end)
        dataSet.D_sample[space] = data_info.data_ingestion(data_info.sample_start, data_info.sample_end)
        
        # Data preprocessing based on purposes
        #D
        dataSet.D_merged[space] = dataSet.get_mergedD(dataSet.D_original[space], data_info.resample_min, data_info.drop_features)
        #D_clean
        dataSet.D_clean[space]  = dataSet.get_cleanD(dataSet.D_original[space], dataSet.D_sample[space], data_info.data_range, data_info.resample_min, data_info.limit_minute_for_nan_processing)
        #dqch.check_data_quality(D_clean[space])    
        #D_extended
        dataSet.D_extended[space] = dataSet.get_extendedD(dataSet.D_clean[space], data_info.data_range_level, data_info.vector_feature_list)
        #D_selected
        dataSet.D_selected[space] = dataSet.get_selectedD(dataSet.D_extended[space])  
        #D_scaled
        dataSet.D_scaler[space], dataSet.D_scaled[space] = dataSet.get_scaledD(dataSet.D_selected[space])
       
    return dataSet

def data_up_sampling(data, freq):
    data = data.sort_index(axis=1)
    data_up = data.resample(freq, label='left').mean()
    data_up = data_up.interpolate(method='values')
    return data_up
    
def data_down_sampling(data, freq):
    data = data.sort_index(axis=1)
    data_down = data.resample(freq, label='left').mean()
    return data_down

### Plot 
import matplotlib.pyplot as plt

def plot_all_feature_data(data):
    plot_cols = data.columns
    plot_features = data[plot_cols]
    _ = plot_features.plot(subplots=True)


def plot_all_feature_data_one_pic(data):
    data.plot()
    plt.legend()
    plt.show()
    
def plot_all_feature_data_two_columns(data):

    import math
    fig, axes = plt.subplots(nrows=int(math.ceil(len(data.columns)/2)), ncols=2, dpi=130, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        if i<len(data.columns):
            temp = data[data.columns[i]]
            ax.plot(temp, color='red', linewidth=1)
            ax.set_title(data.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines['top'].set_alpha(0)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
 

def plot_correlation_chart(data):
    import seaborn as sns
    corr = data.corr()
    ax = sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values, annot =True, annot_kws ={'size': 8})
    bottom, top = ax.get_ylim() 
    heat_map = plt.gcf()
    ax.set_ylim(bottom+0.5, top-0.5)
    heat_map.set_size_inches(10, 6)
    plt.show()
    

    