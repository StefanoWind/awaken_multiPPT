# -*- coding: utf-8 -*-
"""
Optimize scan trajectory in front of AWAKEN turbines
"""

import os
cd=os.path.dirname(__file__)
import xarray as xr
import utm
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import minimize
import utils as utl

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#dataset
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbine_sel=['H04','H05','H06']
h=89#[m] hub height
d=127#[m] rotor diameter
lidar_lat=36.362114#[deg] latitude
lidar_lon=-97.40512#[deg] longitude
h_lidar=0.5#[m] height of the lidar a.g.l.
turb_z={'H04':0,'H05':-1,'H06':-5}#[m] relative altitude of tower base vs lidar base

#optimizer
N_opt=20#number of optimizations
center=[0,-3*d]#[m] center of scan volume
rmax=d#[m] radius of scan volume
Nb=6#number of beams
weight=0.5#weight of wind speed vs. wind speed variance
L=10000#nuber of MC draws for check

#graphics
colors={'H04':'r','H05':'g','H06':'b'}

#%% Functions

def error(theta_r,x0,y0,z0,weight):
    
    #extract scan geometry
    Nb=int(len(theta_r)/2)
    theta=theta_r[:Nb]
    r=theta_r[Nb:]
    
    azi,ele=utl.pol2spher(theta,r,x0,y0,z0)
    assert len(azi)==len(ele), "Lenghts of azimuth and elevation mismatch"
    
    Nb=len(azi)
    beta=ele
    alpha=(90-azi)%360
    
    #calculate errors
    var_u=utl.error_ws(alpha, beta)
    var_uu=utl.error_uu(alpha, beta)
    
    return var_u*weight+var_uu*(1-weight)

#%% Initialization
turbines=xr.open_dataset(source_layout,group='turbines')

lidar_x=utm.from_latlon(lidar_lat,lidar_lon)[0]
lidar_y=utm.from_latlon(lidar_lat,lidar_lon)[1]

os.makedirs(os.path.join(cd,'data','scans'),exist_ok=True)

turb_x={}
turb_y={}
for t in turbine_sel:
    turb_x[t]=turbines.x_utm[turbines.name==t].values[0]-lidar_x
    turb_y[t]=turbines.y_utm[turbines.name==t].values[0]-lidar_y
    
#graphics
fig=plt.figure(figsize=(18,8))
ax = fig.add_subplot(111, projection='3d')

#%% Main

for t in turbine_sel:
    
    #scan optimization
    f_opt=[]
    azi_opt=[]
    ele_opt=[]
    f_best=[]
    azi_best=[]
    ele_best=[]

    while len(f_opt)<N_opt:
        
        #location
        x0=turb_x[t]+center[0]
        y0=turb_y[t]+center[1]
        z0=turb_z[t]+h-h_lidar
            
        #initial point
        theta0=np.random.rand(Nb)*360
        rmax0=np.random.rand(Nb)*rmax
         
        #optimize
        res = minimize(error, np.append(theta0,rmax0), method='SLSQP', tol=1e-7,
                       bounds=([(0,359.99)]*Nb+[(0,rmax)]*Nb),
                       options={'maxiter':1000},args=(x0,y0,z0,weight))
        
        if res.success==True: 
            theta_opt=res.x[:Nb]
            r_opt=res.x[Nb:]
            
            azi_opt.append(np.round(utl.pol2spher(theta_opt,r_opt,x0,y0,z0)[0],2))
            ele_opt.append(np.round(utl.pol2spher(theta_opt,r_opt,x0,y0,z0)[1],2))
            f_opt.append(np.around(res.fun,5))
        
            f_best.append(min(f_opt))
            azi_best.append(azi_opt[np.where(f_opt==f_best[-1])[0][0]])
            ele_best.append(ele_opt[np.where(f_opt==f_best[-1])[0][0]])
        else:
            print(res.message)
        print(t+': '+str(len(f_opt)))
        
    #MC check
    f=np.zeros(L)
    for i in range(L):
        theta0=np.random.rand(Nb)*360
        rmax0=np.random.rand(Nb)*rmax
        f[i]=error(np.append(theta0,rmax0),x0,y0,z0,weight)
        
    print(f'MC optimization beats SLSQP {np.sum(f<f_best[-1])} times out of {L}')

    #output
    output=pd.DataFrame()
    output['Azimuth']=azi_best[-1]
    output['Elevation']=ele_best[-1]
    output.sort_values(by='Azimuth').to_excel(os.path.join(cd,'data','scans',f'{t}.w{weight}.o{N_opt}.xlsx'),index=False)
    
    #plot
    for a,e in zip(azi_best[-1],ele_best[-1]):
        rh=z0/utl.sind(e)
        plt.plot([0,rh*utl.cosd(e)*utl.cosd(90-a)],[0,rh*utl.cosd(e)*utl.sind(90-a)],[h_lidar,h_lidar+rh*utl.sind(e)],color=colors[t])
    plt.plot(x0+utl.cosd(np.arange(360))*rmax,y0+utl.sind(np.arange(360))*rmax,np.arange(360)*0+z0,color=colors[t],linewidth=1)

    utl.draw_turbine_3d(ax,turb_x[t],turb_y[t],turb_z[t]+h,d,h,180)

#%% Plots
ax.set_xlim([-750,750])
ax.set_ylim([-250,350])
ax.set_zlim([0,h+d])
ax.set_aspect('equal')
ax.set_xlabel('W-E [m]')
ax.set_ylabel('S-N [m]')
ax.set_zlabel(r'$z$ [m a.g.l.]')



