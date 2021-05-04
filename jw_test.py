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
        self.data_range = {'max_num':{'CO2ppm':10000, 'H2Sppm':100, 'NH3ppm':300, 'OptimalTemperature':45, 
                               'RichTemperature':45, 'RichHumidity':100, 'comp_temp':45, 'comp_humid':100,
                               '2ndPanSpeed':200, 'UpperPanSpeed':200,
                              'out_humid':100,'out_pressure':2000,'out_temp':50 },
                   'min_num':{'CO2ppm':0, 'H2Sppm':0, 'NH3ppm':0, 'OptimalTemperature':-20, 'RichTemperature':-20, 
                              'RichHumidity':0, 'comp_temp':-20, 'comp_humid':0, '2ndPanSpeed':0, 'UpperPanSpeed':0,
                             'out_humid':0, 'out_pressure':0, 'out_temp':-30}}
    
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
                 print(data_raw[data_char])
                 #ifm.Influx_management(config.host_, config.port_, config.user_, config.pass_, db_name, config.protocol, table=table_name).get_raw_data(start, end)

                 
        return data_raw
    

if __name__ == "__main__":
    start='2020-09-30 00:00:01'
    end ='2020-10-18 00:00:00'
    intDataInfo = {
            "Data":[{"db_name":"INNER_AIR", "measurement":"HS1", "domain":"farm", "subdomain":"airQuality"},
                    {"db_name":"OUTDOOR_AIR", "measurement":"sangju", "domain":"city", "subdomain":"airQuality" },
                    {"db_name":"OUTDOOR_WEATHER", "measurement":"sangju", "domain":"city", "subdomain":"weather"}],
            "start":str(start),
            "end":str(end)
    }
    from KETI_setting import influx_setting_KETI as ins
    from data_influx import ingestion_measurement as ing
    
    DataList = intDataInfo["Data"]
    start = intDataInfo['start']
    end = intDataInfo['end']
    result={}
    for i, dbinfo in enumerate(DataList):
        db_name = dbinfo['db_name']
        measurement = dbinfo['measurement']
        influx_c = ing.Influx_management(ins.host_, ins.port_, ins.user_, ins.pass_, db_name, ins.protocol)
        result[i] = influx_c.get_df_by_time(start,end,measurement)
        print(result[i])
        
