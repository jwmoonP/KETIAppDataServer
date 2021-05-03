import numpy as np 

def data_range_error_treatment(data, sample, data_range):
    data_out = data.copy()
    data_out = data_range_trimming(data_out, data_range)
    data_out = error_detection_with_neighbor(sample, data_out, data_range)
    data.index.name='datetime'
    data_out.index.name='datetime'
    return data_out

## for server
def error_detection_with_neighbor(sample, data, sensor_limit):
    #first_ratio=0.05
    first_ratio=0.5
    second_ratio=0.5
    
    data_out = data.copy()
    for feature in data_out.columns:
        test_data= data_out[[feature]].copy()
        sample_data = sample[[feature]].copy()
        # test_data = outlier_extream_value_analysis(sample_data, test_data, 10) (나중에 수정해서 필요시 사용해야 함)
        test_data= outlier_detection_two_step_neighbor(sample_data, test_data, first_ratio, second_ratio)
        data_out[[feature]] = test_data  
    return data_out
## for server

def data_range_trimming(data, sensor_limit):
    column_list = data.columns
    max_list = sensor_limit['max_num']
    min_list = sensor_limit['min_num']
    
    for column_name in column_list:
        if column_name in max_list.keys():  
            max_num = max_list[column_name]
            min_num = min_list[column_name]
            
            mask = data[column_name] > max_num
            #merged_result.loc[mask, column_name] = max_num
            data[column_name][mask] = np.nan#max_num
            mask = data[column_name] < min_num
            #merged_result.loc[mask, column_name] = min_num
            data[column_name][mask] = np.nan#min_num
        
    return data
    
        
def outlier_detection_two_step_neighbor(sample, data, first_ratio, second_ratio):
    column_list = data.columns
    data_out1 = data.copy()
    for column_name in column_list:
        temp = data_out1[[column_name]]
        sample_temp = sample[[column_name]]
        data_1 = temp.diff().abs()
        data_2 = temp.diff(periods=2).abs()
        temp_mean = sample_temp.mean().values[0]
        First_gap = temp_mean* first_ratio
        Second_gap = temp_mean * second_ratio
        data_1_index = data_1[data_1[column_name] > First_gap].index.tolist()
        data_2_index = data_2[data_2[column_name] < Second_gap].index.tolist()
        noise_index = set(data_1_index)&set(data_2_index)

        for noise in noise_index:
            pos = data_out1.index.get_loc(noise)
            data_out1.iloc[pos-1].loc[column_name] = data_out1.iloc[pos].loc[column_name]
    return data_out1

def outlier_extream_value_analysis(sample, data, extream):
    data_out= data.copy()
    for feature in data_out.columns:
        test_data = data_out[[feature]].copy()
        sample_data = sample[[feature]].copy()
        IQR = sample_data.quantile(0.75)-sample_data.quantile(0.25)
        upper_limit = sample.quantile(0.75)+(IQR*1.5)
        upper_limit_extream = sample_data.quantile(0.75)+(IQR*extream)
        lower_limit = sample_data.quantile(0.25)-(IQR*1.5)
        lower_limit_extream = sample_data.quantile(0.25)-(IQR*extream)
        test_data[(test_data < lower_limit_extream)] =np.nan
        test_data[(test_data> upper_limit_extream)] =np.nan
        
        test_data = test_data.fillna(method='ffill', limit=1)
        test_data = test_data.fillna(method='bfill', limit=1)
        data_out[[feature]] = test_data
    
    return data_out