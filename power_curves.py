# -*- coding: utf-8 -*-
'''
Power curves from titled profiling
'''
import os
cd=os.path.dirname(__file__)
import sys
import warnings
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
import re
import matplotlib.dates as mdates
from scipy import interpolate
import glob
import xarray as xr
import numpy as np

warnings.filterwarnings('ignore')

#%% Inputs
path='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/AWAKEN/awaken_multiPPT/data/awaken/sa1.lidar.z07.c1'
source_power=os.path.join(cd,'data/awaken/kp.turbine.z01.a0/*nc')
flags=['h04','h05','h06']
H=90
D=127
max_rmsd=0.5
colors={'h04':'r','h05':'g','h06':'b'}

#%% Functions
def nan_corr(a,b):
    real=~np.isnan(a+b)
    return np.corrcoef(a[real],b[real])[0,1]

#%% Initialization
files_power=glob.glob(source_power)
scada=xr.open_mfdataset(files_power,combine='nested',concat_dim='time')

ws={}
se={}
rmsd={}
power={}

#%% Main
for flag in flags:
    files=glob.glob(os.path.join(path,f'*{flag}*.nc'))
    data=xr.open_mfdataset(files,combine='nested',concat_dim='time')
    ws_all=[]
    se_all=[]
    rmsd_all=[]
    power_all=[]
    for i in range(len(data.time)):
        ws_sel=data.WS.isel(time=i).sel(height=slice(H-D/2, H+D/2))
        real=~np.isnan(ws_sel.values)
        try:
            LF=np.polyfit(np.log(ws_sel.height[real]/H),np.log(ws_sel.values[real]),1)
            ws_all=np.append(ws_all,np.exp(LF[1]))
            se_all=np.append(se_all,LF[0])
            rmsd_all=np.append(rmsd_all,np.nanmean((ws_sel.values-ws_all[-1]*(ws_sel.height/H)**se_all[-1])**2)**0.5)
        except:
            ws_all=np.append(ws_all,np.nan)
            se_all=np.append(se_all,np.nan)
            rmsd_all=np.append(rmsd_all,np.nan)
        
        t1=data.start_time.isel(time=i).values
        t2=data.end_time.isel(time=i).values
        power_all=np.append(power_all,scada.power.sel(t_id=flag,time=slice(t1,t2)).mean().values)
        
        
    ws[flag]=xr.DataArray(data=ws_all,coords={'time':data.time.values})
    se[flag]=xr.DataArray(data=se_all,coords={'time':data.time.values})
    rmsd[flag]=xr.DataArray(data=rmsd_all,coords={'time':data.time.values})
    power[flag] =xr.DataArray(data=power_all,coords={'time':data.time.values})
    
    ws[flag]= ws[flag].where(rmsd[flag]<max_rmsd)
    power[flag]=power[flag].where(power[flag]>0)
    
#%% Plots
plt.figure(figsize=(18,8))
for flag in flags:
    plt.subplot(4,1,1)
    plt.plot(ws[flag].time,ws[flag].values,'.',label=flag,color=colors[flag])
    plt.ylabel(r'$\overline{u}(H)$ [m s$^{-1}$]')
    plt.grid()
    
    plt.subplot(4,1,2)
    plt.plot(se[flag].time,se[flag].values,'.',label=flag,color=colors[flag])
    plt.ylabel(r'$\alpha$')
    plt.grid()
    
    plt.subplot(4,1,3)
    plt.plot(rmsd[flag].time,rmsd[flag].values,'.',label=flag,color=colors[flag])
    plt.ylabel(r'RMSD$(\overline{u}$) [m s$^{-1}$]')
    plt.grid()
    
    plt.subplot(4,1,4)
    plt.plot(power[flag].time,power[flag].values,'.',label=flag,color=colors[flag])
    plt.ylabel(r'$P$ [kW]')
    plt.grid()
    
plt.legend()

plt.figure(figsize=(18,4))
for flag in flags:
    plt.scatter(ws[flag],power[flag],s=rmsd[flag]*10+5,label=flag,color=colors[flag])
    plt.xlabel(r'$\overline{u}(H)$ [m s$^{-1}$]')
    plt.ylabel(r'$P$ [kW]')
    plt.grid()
   
plt.legend()