'''
Created on Apr 23, 2019

@author: moonjaewon
'''

import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, session, request, g, send_file, redirect, flash
from flask_bootstrap import Bootstrap

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)
app.secret_key = "super secret key"

start='2020-09-30 00:00:01'
end ='2020-10-18 00:00:00'
########################################## Server feature
visual_resample_min='60min'
########################################## Template 
#g.start, g.end
###########################################

@app.route('/',  methods=['GET'])
def index():
    import data_selection.static_dataSet as sDS
    intDataInfo = sDS.set_integratedDataInfo(start, end)
    result = get_data(intDataInfo)
    return render_template('/index.html')

def get_data(intDataInfo):
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
    
    return result


###################### Server 3

if __name__ == '__main__':
    app.run(debug=True, port='9008')
    


