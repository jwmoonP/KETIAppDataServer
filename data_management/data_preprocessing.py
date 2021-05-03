import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd 
import math 
import data_management.data_outlier as do

class ProcessedData():
    def __init__(self):
        pass
        
    def data_integration(self, data, resample_min):
        resample_freq = str(resample_min)+'min'
        data_key_list = list(data.keys())
        merged_data_list =[]
        data_resample={}
        for data_name in data_key_list:
            data_resample[data_name] = data[data_name].resample(resample_freq, label='left').mean()
            data_resample[data_name] = data_resample[data_name].fillna(method='bfill', limit=1)
            data_resample[data_name] = data_resample[data_name].fillna(method='ffill', limit=1)
            merged_data_list.append(data_resample[data_name])
        
        merged_data = pd.concat(merged_data_list, axis=1, join='inner')
        merged_data=merged_data.sort_index(axis=1)
       
        return merged_data
        

    def data_validation(self, data):
        data = data.loc[:, ~data.columns.duplicated()]
        first_idx = data.first_valid_index()
        last_idx = data.last_valid_index()
        valid_data = data.loc[first_idx:last_idx]
        
        return valid_data
    
    def data_cleaning(self, data_original, data_sample, data_range):
            data_key_list = list(data_original.keys())
            data_pre={}
            #feature_list={}
            for data_name in data_key_list:
                data_pre[data_name] = do.data_range_error_treatment(data_original[data_name], data_sample[data_name], data_range)

            return data_pre
    
    def data_nan_processing(self, data, resample_min, limit_minute_for_nan_processing): 
        data_key_list = list(data.keys())
        limit_num_for_nan_processing = math.ceil(limit_minute_for_nan_processing/resample_min)  
        data_non_nan= data.interpolate(method='values', limit= limit_num_for_nan_processing)
        return data_non_nan
        
    
    def data_scaling(self, data, scaler =MinMaxScaler(feature_range=(0, 1))):
         #scaling
        scaler =MinMaxScaler(feature_range=(0, 1))
        data_scale = pd.DataFrame(scaler.fit_transform(data), index=data.index,columns=data.columns)
        return scaler, data_scale

"""
class DataPreProcessing():
    
    def _get_how_dict(self, float_column, nofloat_column):

        how_dict = {}
        for key in float_column:
            how_dict[key]='mean'
            
        for key in nofloat_column:
            how_dict[key]=self._frequent
            
        return how_dict
    
      
    def data_sampling(self, data, resample_min):
        float_column, nofloat_column = get_column_list(data)
        how_dict = self._get_how_dict(float_column, nofloat_column)
        #data[resample_column]=data[resample_column].resample(resample_min).mean()
        #data[noresample_column]=data[noresample_column].resample(resample_min, closed='left').mode().astype('unit8')
        data = data.resample(resample_min).agg(how_dict)
        #data[noresample_column]=data[noresample_column].resample(resample_min).agg(how_dict)astype('unit8')
        
        return data
    
    def _frequent(self, x):
        if len(x) ==0 : return -1
        (value, counts) = np.unique(x, return_counts=True)
        return value[counts.argmax()]

    
    def data_outlier_remove(self, data, sensor_limit):
        column_list = data.columns
        
        for column_name in column_list:
            max_list = sensor_limit['max_num']
            min_list = sensor_limit['min_num']
            max_num = max_list[column_name]
            min_num = min_list[column_name]
            
            mask = data[column_name] > max_num
            #merged_result.loc[mask, column_name] = max_num
            data[column_name][mask] = max_num
            mask = data[column_name] < min_num
            #merged_result.loc[mask, column_name] = min_num
            data[column_name][mask] = min_num
            
        return data
        
    def data_smoothing(self, data, rolling_time):
        float_column, nofloat_column = get_column_list(data)

        if rolling_time !='0':
            data[float_column]=data[float_column].rolling(rolling_time).mean()
       
        return data
    
    def data_round(self, data, round_digit):
        float_column, nofloat_column = get_column_list(data)
        data[float_column]=data[float_column].round(round_digit)
        return data
       
    
    def data_log_scaling(self, data, log_cloumn_list):
        #precondition: All data types are number
        #log_cloumn_list = list(data.select_dtypes(include=['float']).columns)
        data[log_cloumn_list] = np.log(data[log_cloumn_list])
        return data
        
        
class data_preprocessing():
    def __init__(self, rolling_time, round_digit):
        self.rolling_time = rolling_time
        self.round_digit = round_digit
    
    def get_preprocessing_data(self, data):
        dp = DataPreProcessing()
        # integration 이 retrieval 안에 
        data = dp.data_smoothing(data, self.rolling_time)

        #Feature Engineering
        #fe = pre.FeatureEngineering()
        #data = fe.data_feature_engineering_air(data)
        #data = fe.feature_selection(data, self.feature_list)
        
        # Data Lighweight
        data = dp.data_round(data, self.round_digit)
    
        return data 
    
"""    

