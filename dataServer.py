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
########### 
################################
data_raw_partial={}
@app.route('/',  methods=['GET'])
def index():
    from data_selection import static_dataSet 
    from KETI_pre_dataIngestion.data_influx import ingestion_partial_dataset as ipd
    from KETI_pre_dataIngestion.KETI_setting import influx_setting_KETI as ins
    #from KETI_pre_dataCleaning.extream_data import ingestion_partial_dataset as ipd
    # data selection
    intDataInfo = static_dataSet.set_integratedDataInfo(start, end)
    # data ingestion
    data_raw_partial = ipd.partial_dataSet_ingestion(intDataInfo, ins)
    # data cleaning
    #data_clean_partial = 
    
    return render_template('/index.html')

 
    



###################### Server 3

if __name__ == '__main__':
    app.run(debug=True, port='9008')
    


