# -*- coding: utf-8 -*-
"""
Extract power data from 50-Hz QC'ed load data
"""

import os
cd=os.path.dirname(__file__)
import sys
import warnings
import yaml
import matplotlib.pyplot as plt
import re
from datetime import datetime
import matplotlib.dates as mdates
from scipy import interpolate
import pandas as pd
import glob
import xarray as xr
import numpy as np


#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-06-03' #start date
    edate='2025-06-04' #end date
    path_config=os.path.join(cd,'configs/config_235.yaml') #config path
else:
    sdate=sys.argv[1] #start date
    edate=sys.argv[2]  #end date
    path_config=sys.argv[3]#config path
    
    
#%% Initialization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

columns=[f'PKGP1HIST01.OKWF001_KP_Turbine{t.replace("0","").upper()}.ActivePower' for t in config['turbine_flags']]


#%% Main
#download files

for channel in config['channels_scada']:
    folder=os.path.join(config['path_data'],channel.replace('00','a0'))
    os.makedirs(folder,exist_ok=True)
    
    files=np.array(sorted(glob.glob(os.path.join(config['path_data'],channel,'*.parquet'))))
    t_files=[]
    for file in files:
        match = re.search(r'\d{8}\.\d{6}', file)
        t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
        t_files=np.append(t_files,t)

    sel_t=(t_files>=datetime.strptime(sdate,'%Y-%m-%d'))*(t_files<datetime.strptime(edate,'%Y-%m-%d'))

    for file,t_file in zip(files[sel_t],t_files[sel_t]):
        filename=f'{channel.replace("00","a0").split("/")[1]}.{datetime.strftime(t_file,"%Y%m%d.%H%M%S")}.nc'
        
        temp = pd.read_parquet(file).astype(np.float64)
        
        # Convert to numpy.datetime64
        time = temp.index.tz_convert(None).values
        
        #convert to xarray
        data=xr.Dataset()
        data['power']=xr.DataArray(data=temp[columns].values,coords={'time':time,'t_id': config['turbine_flags']})
        
        #save output
        data.to_netcdf(os.path.join(folder,filename))
    
    
