# -*- coding: utf-8 -*-
'''
Calculate wind profiled from six-beam
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
import pandas as pd

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2025-06-03' #start date
    edate='2025-06-03' #end date
    replace=False #replace existing files
    path_config=os.path.join(cd,'configs/config_235.yaml') #config path
    mode='serial' #processing mofe (serial or parallel)
else:
    sdate=sys.argv[1]
    edate=sys.argv[2]
    replace=sys.argv[3]=="True"
    path_config=sys.argv[4]
    mode=sys.argv[5]
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#list of days to process
days=np.arange(np.datetime64(sdate+'T00:00:00'),np.datetime64(edate+'T00:00:00')+np.timedelta64(1,'D'),np.timedelta64(1,'D'))

#%% Functions
def cosd(x):
    return np.cos(x/180*np.pi)

def sind(x):
    return np.sin(x/180*np.pi)

def general_inv(A):
    try:
        A_inv=np.linalg.inv(A)
    except:
        A_inv=np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A))
        
    return A_inv

def vstack(a,b):
    '''
    Stack vertically vectors
    '''
    if len(a)>0:
        ab=np.vstack((a,b))
    else:
        ab=b
    return ab   

def fill_only_short_nans(arr, limit=3):
    output=arr.copy()
    if np.sum(~np.isnan(arr))>1:
        sel=np.arange(np.where(~np.isnan(arr))[0][0],np.where(~np.isnan(arr))[0][-1]+1)
        s = pd.Series(arr[sel])
        is_nan = s.isna()
        
        # Identify contiguous NaN blocks
        nan_groups = (is_nan != is_nan.shift()).cumsum()[is_nan]
        counts = nan_groups.value_counts()
    
        # Mask: True where NaN group is longer than limit
        long_nan_mask = is_nan.copy()
        for group_id, count in counts.items():
            if count > limit:
                long_nan_mask[nan_groups[nan_groups == group_id].index] = False
    
        # Replace long NaNs with a placeholder (e.g., remain NaN)
        temp = s.copy()
        temp[~long_nan_mask & is_nan] = np.nan  # explicitly mark long nans again
    
        # Interpolate only short NaNs
        temp = temp.interpolate(limit=limit, limit_direction="both")
    
        # Restore long NaNs
        temp[~long_nan_mask & is_nan] = np.nan
        
        output[sel]=temp.to_numpy()

    return output


def wind_retrieval(files,config,lidar_height,save_path, replace):
    
    if len(files)==0:
        return

    #zeroing
    U=[]
    V=[]
    W=[] 
    uu=[]
    vv=[]
    ww=[]
    uv=[]
    uw=[]
    vw=[]
    
    time=np.array([],dtype='datetime64')
    stime=np.array([],dtype='datetime64')
    etime=np.array([],dtype='datetime64')
    
    #file naming
    files=sorted(files)
    match = re.match(r"^(.*\D)(\d{8}\.\d{6})(.*)$", os.path.basename(files[0]))
    filename=f'{match.group(1)}{str(match.group(2))[:8]}.000000{match.group(3)}'.replace('b0','c1')
    
    if save_path==None:
        save_path=os.path.dirname(files[0]).replace('b0','c1')
    os.makedirs(save_path,exist_ok=True)
    
    #initialize logger
    logfile=os.path.join(cd,'log',filename.replace('nc','log'))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    os.makedirs('log',exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if os.path.isfile(os.path.join(save_path,filename)) and replace==False:
        logging.info(f'File {filename} already exists, skipping')
        return
    
    for f in files:
        logging.info(f'Processing {os.path.basename(f)}')
        data=xr.open_dataset(f)
        Nb=len(data.beamID)
        
        #cartesian angles
        alpha=(90-data.azimuth.mean(dim='scanID').values)%360
        beta=data.elevation.mean(dim='scanID').values
        
        #time
        time_srt=data.time.isel(scanID=0,beamID=0)
        time_end=data.time.isel(scanID=-1,beamID=-1)
        time_avg=time_srt+(time_end-time_srt)/2
        
        #average and interpolate RWS
        z=data.z.values+lidar_height
        rws_avg=data.wind_speed.where(data.qc_wind_speed==0).mean(dim='scanID').values
        height=np.arange(config['min_height'],config['max_height']+config['height_step'],config['height_step'])
        
        rws_avg_int=np.zeros((len(height),Nb))
        for i in range(Nb):
            x=np.concatenate([[0],z[:,i],[np.max([z[-1,i],config['max_height']])+1]])
            y=fill_only_short_nans(np.concatenate([[np.nan],rws_avg[:,i],[np.nan]]),limit=config['max_nans'])
            f_int=interpolate.interp1d(x,y, kind='linear')
            rws_avg_int[:,i]=f_int(height).T
    
        #velocity vector
        A=[]
        for a,b in zip(alpha,beta):
            A=vstack(A,np.array([cosd(b)*cosd(a),cosd(b)*sind(a),sind(b)]))
        A_inv=general_inv(A)
            
        vel_vector=A_inv@rws_avg_int.T
        
        #store wind velocity
        U=vstack(U,vel_vector[0,:])
        V=vstack(V,vel_vector[1,:])
        W=vstack(W,vel_vector[2,:])
        time=np.append(time,time_avg)
        stime=np.append(stime,time_srt)
        etime=np.append(etime,time_end)
        
        #variance
        rws_var=data.wind_speed.where(data.qc_wind_speed==0).var(dim='scanID').values
        
        rws_var_int=np.zeros((len(height),Nb))
        for i in range(Nb):
            x=np.concatenate([[0],z[:,i],[np.max([z[-1,i],config['max_height']])+1]])
            y=fill_only_short_nans(np.concatenate([[np.nan],rws_var[:,i],[np.nan]]),limit=config['max_nans'])
            f_int=interpolate.interp1d(x,y, kind='linear')
            rws_var_int[:,i]=f_int(height).T
            
        #reynolds stresses
        A=[]
        for a,b in zip(alpha,beta):
            A=vstack(A,
            [cosd(b)**2*cosd(a)**2,
             cosd(b)**2*sind(a)**2, 
             sind(b)**2,
             2*cosd(b)**2*cosd(a)*sind(a),  
             2*cosd(b)*sind(b)*cosd(a),
             2*cosd(b)*sind(b)*sind(a)])               
        
        RS=np.matmul(general_inv(A),rws_var_int.T)
            
        uu=vstack(uu,RS[0,:])
        vv=vstack(vv,RS[1,:])
        ww=vstack(ww,RS[2,:])
        uv=vstack(uv,RS[3,:])
        uw=vstack(uw,RS[4,:])
        vw=vstack(vw,RS[5,:])
        
    #output
    Output=xr.Dataset()
    Output['start_time']=xr.DataArray(data=stime,coords={'time':time})
    Output['end_time']=xr.DataArray(data=etime,coords={'time':time})
    Output['U']=xr.DataArray(data=U,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average W-E wind component'})
    Output['V']=xr.DataArray(data=V,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average S-N wind component'})
    Output['W']=xr.DataArray(data=W,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average vertical wind component'})
    
    Output['WS']=(Output['U']**2+Output['V']**2)**0.5
    Output['WS'].attrs={'units':'m/s','description':'average horizontal wind speed'}
    
    Output['WD']=(270-np.degrees(np.arctan2(Output['V'],Output['U'])))%360
    Output['WD'].attrs={'units':'degrees','description':'average horizontal wind direction (0=N, 90=E)'}
    
    Output['uu']=xr.DataArray(data=uu,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'W-E velocity variance'})
    Output['vv']=xr.DataArray(data=vv,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'S-N velocity variance'})
    Output['ww']=xr.DataArray(data=ww,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical velocity variance'})
    Output['uv']=xr.DataArray(data=uv,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'horizontal (W-E to S-N) Reynolds stress'})
    Output['uw']=xr.DataArray(data=uw,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical Reynolds stress in W-E direction'})
    Output['vw']=xr.DataArray(data=vw,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical Reynolds stress in S-N direction'})
    
    Output['tke']=xr.DataArray(data=(uu+vv+ww)/2,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'turbulence kintic energy'})
    
    uu_rot= Output['uu']*cosd(270-Output['WD'])**2+2*Output['uv']*cosd(270-Output['WD'])*sind(270-Output['WD'])+Output['vv']*sind(270-Output['WD'])**2
    Output['ti']=uu_rot**0.5/Output['WS']*100
    Output['ti'].attrs={'units':'%','description':'streamwise turbulence intensity'}
    
    Output['u_star']=(Output['uw']**2+Output['vw']**2)**0.25
    Output['u_star'].attrs={'units':'m/s','description':'friction velocity'}
    
    logging.info(f'Wind profiles saves as {os.path.join(save_path,filename)}')
    Output.to_netcdf(os.path.join(save_path,filename))
    
    #plots
    wind_map(Output,os.path.join(save_path,filename))
    
    return Output

    
def wind_map(data,filename):
    
    barb_stagger_time = 1#skipped time samples in barbs
    barb_stagger_height = 10#skipped height samples in barbs
    colorbar_fs = 14#colorbar fontsize
    label_fs = 14#colorbar fontsize
    tick_fs = 14#colorbar fontsize
    
    date=str(np.min(data.time.values))[:10]
    dtime=int(np.round(np.median(np.diff(data.time))/np.timedelta64(1,'m')))
    offset=int(np.round((np.min(data.time)-np.datetime64(date+'T00:00:00'))/np.timedelta64(1,'m')))
    data=data.resample(time=f'{dtime}min',offset=f'{offset-1}min').nearest(tolerance='2min')
    data['time']=data.time+np.timedelta64(1,'m')
    
    fig=plt.figure(figsize=(18,10))
    ax=plt.subplot(2,1,1)
    CS = ax.contourf(data.time, data.height, data.WS.T, np.round(np.arange(np.nanpercentile(data.WS,5), np.nanpercentile(data.WS,95)+0.5, 0.25),1), extend='both', cmap='coolwarm')
    ax.barbs(data.time[::barb_stagger_time], data.height[::barb_stagger_height], data.U.T[::barb_stagger_height,::barb_stagger_time]*1.94, data.V.T[::barb_stagger_height,::barb_stagger_time]*1.94,
        barbcolor='black', flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1)
    ax.barbs(data.time[0]+np.timedelta64(1260,'s'), 3600, 10*np.cos(60), -10*np.sin(60), barbcolor='black',
        flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1.4)
    ax.text(data.time[0]+np.timedelta64(600,'s'), 3480, '10 kts \n', fontsize=12, bbox=dict(facecolor='none', edgecolor='black', alpha=0.8))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '2%', pad=0.65)
    cb = fig.colorbar(CS, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=colorbar_fs)
    cb.set_label(r'Mean wind speed [m s$^{-1}$]', fontsize=colorbar_fs)
    ax.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax.set_ylabel(r'z [m.a.g.l]', fontsize=label_fs)
    ax.set_xlim(data.time.min()-np.timedelta64(300,'s'),data.time.max()+np.timedelta64(900,'s'))
    ax.set_ylim(0, np.max(data.height))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(str(data.time.values[0])[:10], fontsize=label_fs)

    ax=plt.subplot(2,1,2)
    CS = ax.contourf(data.time, data.height, data.ti.T, np.round(np.arange(np.nanpercentile(data.ti,5), np.nanpercentile(data.ti,95)+0.5, 1),1), extend='both', cmap='coolwarm')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '2%', pad=0.65)
    cb = fig.colorbar(CS, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=colorbar_fs)
    cb.set_label(r'Turbulent intensity [%]', fontsize=colorbar_fs)
    
    ax.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax.set_ylabel(r'z [m.a.g.l]', fontsize=label_fs)
    ax.set_xlim(data.time.min()-np.timedelta64(300,'s'),data.time.max()+np.timedelta64(900,'s'))
    ax.set_ylim(0, np.max(data.height))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.savefig(filename.replace('.nc','.png'))
    plt.close()

#%% Main
for channel in config['channels_six_beam']:
    save_path=os.path.join(config['path_data'],channel.replace('b0','c1'))
    os.makedirs(save_path,exist_ok=True)
    
    for flag in config['turbine_flags']:
        files={}
        for d in days:
            files[d]=glob.glob(os.path.join(config['path_data'],channel,f'*{str(d)[:10].replace("-","")}*{flag}.nc'))
            
        if mode=='serial':
            for d in days:
                if len(files[d])>0:
                    Output=wind_retrieval(files[d],config,config['lidar_height'][channel],save_path,replace)
        
        elif mode=='parallel':
            args = [(files[d], config,config['lidar_height'][channel], save_path,replace) for d in days]
            with Pool() as pool:
                pool.starmap(wind_retrieval, args)
        else:
            raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
     