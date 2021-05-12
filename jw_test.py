import numpy as np 
import sys, os
sys.path.append("../")

from data_management import data_manager as dm
import pandas as pd


 
class VIBE:
    
    def __init__(self, space, type='VIBE_1'):

        self.space = space
        self.resample_min =10
        self.limit_minute_for_nan_processing = 30
        self.default_feature='CO2ppm'
        self.drop_features = ['3rdPanSpeed', '4thPanSpeed', 'CoolingPad', 'LowerPanSpeed', 'Quarantine', 'WaterMotor']
        
        self.data_char_list = {'farm':{'INNER_AIR':space}, 'out_air':{"OUTDOOR_AIR":'sangju'}, 'out_weather':{'OUTDOOR_WEATHER':'sangju'}}

        self.data_range_level={'NH3ppm':{"level":[0, 20, 25, 1000000 ], 'label':[0, 1, 2]},
                        'H2Sppm':{"level":[0, 0.5, 10000000 ], 'label':[0, 1]},
                        'CO2ppm':{"level":[0, 2500, 5500, 10000000 ], 'label':[0, 1, 2]},
                        'Humidity':{"level":[0.0, 50.0, 80.0, 100.1], 'label':[1, 0, 2]},
                        'Temperature':{"level":[-50.0, 15.0, 21.0, 25.0, 30.0, 100.0 ], 'label':[0, 1, 2, 3, 4]}
                       }
        self.vector_feature_list = {'out_wind_speed':['out_wind_direction', ['out_wind_x', 'out_wind_y']]}
        
        self.in_feature_list = ['CO2ppm', 'H2Sppm', 'Humidity', 'NH3ppm', 'Temperature']
        self.out_feature_list =[ 'out_CO','out_NO2', 'out_O3', 'out_PM10', 'out_PM25', 'out_SO2', 'out_humid',
               'out_pressure', 'out_rainfall', 'out_sunshine', 'out_temp',
               'out_wind_direction', 'out_wind_speed', 'out_wind_x', 'out_wind_y']
        self.out_feature_list = ['out_PM25', 'out_humid', 'out_pressure', 'out_rainfall', 'out_wind_speed']
        self.out_feature_list_add=['out_O3','out_PM10',]
        #self.dbname_list=['KDS1', 'KDS2', 'HS1', 'HS2']
    
    def _get_sample_time(self,space_name):
        if space_name == 'KDS1':
             sample_start = '2020-07-11'
             sample_end = '2020-07-15'
        elif space_name == 'KDS2':
            sample_start = '2020-07-11'
            sample_end = '2020-07-15'
        else:  # HS1, HS2
            sample_start = '2020-09-30 00:00:00'
            sample_end = '2020-10-01 06:00:00'
        return sample_start, sample_end

    def set_time(self, ingestion_start, ingestion_end):
        self.ingestion_start = ingestion_start
        self.ingestion_end = ingestion_end
        self.sample_start, self.sample_end = self._get_sample_time(self.space)
        
    def data_ingestion(self, start, end):
        data_raw={}
        for data_char in self.data_char_list:
            db_name_list = self.data_char_list[data_char]
            for db_name in db_name_list: # now, only considering one db_name
                 table_name = db_name_list[db_name]
                 print(table_name, db_name)
                 influx_c = mi.Influx_management(isk.host_, isk.port_, isk.user_, isk.pass_, db_name, isk.protocol)
                 data_raw[data_char] = influx_c.get_df_by_time(start,end,table_name) 
                 #ifm.Influx_management(config.host_, config.port_, config.user_, config.pass_, db_name, config.protocol, table=table_name).get_raw_data(start, end)

                 
        return data_raw




def data_nan_processing(self, data, resample_min, limit_minute_for_nan_processing): 
    data_key_list = list(data.keys())
    limit_num_for_nan_processing = math.ceil(limit_minute_for_nan_processing/resample_min)  
    data_non_nan= data.interpolate(method='values', limit= limit_num_for_nan_processing)
    return data_non_nan
        

if __name__ == "__main__":
    start='2020-09-30 00:00:01'
    end ='2020-10-18 00:00:00'
    data_type = 'air'
    
    from data_selection import static_dataSet 
    from KETI_pre_dataIngestion.data_influx import ingestion_partial_dataset as ipd
    from KETI_pre_dataIngestion.KETI_setting import influx_setting_KETI as ins
    #from KETI_pre_dataCleaning.extream_data import ingestion_partial_dataset as ipd
    # data selection
    intDataInfo = static_dataSet.set_integratedDataInfo(start, end)
    # data ingestion
    data_partial_raw_dataS = ipd.partial_dataSet_ingestion(intDataInfo, ins)
    
    from KETI_pre_dataCleaning.definite_error_detection import valid_data, min_max_limit_value, outlier_detection
    data_min_max_limit = min_max_limit_value.MinMaxLmitValueSet(data_type).get_data_min_max_limitSet()
    data_partial_clean = valid_data.make_valid_dataSet(data_partial_raw_dataS, data_min_max_limit)
    
    from KETI_pre_dataCleaning.definite_error_detection import outlier_detection  
    data_partial_clean = outlier_detection.make_neighborErrorDetected_dataSet(data_partial_clean)
    print(data_partial_clean)
    

    
    #data_out = error_detection_with_neighbor(sample, data_out, data_range)
    
    
