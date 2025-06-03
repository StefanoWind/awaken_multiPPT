# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:14:19 2025

@author: sletizia
"""

# -*- coding: utf-8 -*-
'''
Processor of lidar through LIDARGO

Inputs (both hard-coded and available as command line inputs in this order):
    sdate [%Y-%m-%d]: start date in UTC
    edate [%Y-%m-%d]: end date in UTC
    delete [bool]: whether to delete raw data
    path_config: path to general config file
    mode [str]: serial or parallel
'''
import os
cd=os.path.dirname(__file__)
import sys
import traceback
import warnings
import lidargo as lg
from datetime import datetime
import yaml
from doe_dap_dl import DAP
from multiprocessing import Pool
import logging
import re
import glob

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-06-03' #start date
    edate='2025-06-03' #end date
    download=False
    delete=False #delete raw files?
    path_config=os.path.join(cd,'configs/config_235.yaml') #config path
    mode='serial'
else:
    sdate=sys.argv[1] #start date
    edate=sys.argv[2]  #end date
    download=sys.argv[3] #download files?
    delete=sys.argv[4]=="True" #delete raw files?
    path_config=sys.argv[5]#config path
    mode=sys.argv[6]
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)

#WDH
if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_cert_auth(username=config['username'], password=config['password'])

#%% Functions
def standardize_file(file,save_path_stand,config,logfile_main,sdate,edate):
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
            lproc = lg.Standardize(file, config=config['path_config_stand'], verbose=True,logfile=logfile)
            lproc.process_scan(replace=False, save_file=True, save_path=save_path_stand)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')

#%% Main

for channel in config['channels']:
    
    #download files
    if download:
        _filter = {
            'Dataset': channel,
            'date_time': {
                'between': [sdate.replace('-','')+'000000',edate.replace('-','')+'235959']
            },
            'file_type':'nc',
            'ext1': 'user5'
        }
        
        os.makedirs(os.path.join(config['path_data'],channel),exist_ok=True)
        a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)
     
        
    #standardize all files within date range
    files=glob.glob(os.path.join(config['path_data'],channel,'*a0*user5.nc'))
    save_path_stand=os.path.join(config['path_data'],channel.replace('a0','b0'))
    if mode=='serial':
        for f in files:
              standardize_file(f,save_path_stand,config,logfile_main,sdate,edate)
    elif mode=='parallel':
        args = [(files[i],save_path_stand, config,logfile_main,sdate,edate) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(standardize_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
          

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
