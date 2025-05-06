# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:32:28 2025

@author: sletizia
"""

import os
cd=os.path.dirname(__file__)
import xarray as xr
import utm
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import minimize
from utils import *


plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14


#%% Inputs
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbine_sel=['H04','H05','H06']
h=89#[m] hub height
d=127
lidar_lat=36.362114#[deg] latitude
lidar_lon=-97.40512#[deg] longitude
h_lidar=0.5#[m]
turb_z={'H04':0,'H05':-1,'H06':-5}
N_opt=5

center=[0,-3*d]
rmax=d
Nb=6
weight=0.99

L=10000

colors={'H04':'r','H05':'g','H06':'b'}

#%% Functions

def error_ws(alpha,beta):
    #cartesian->LOS matrix
    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([cosd(beta[i])*cosd(alpha[i]),cosd(beta[i])*sind(alpha[i]),sind(beta[i])])

    #LOS->cartesian matrix
    try:
        A_plus=np.linalg.inv(A.T@A)@A.T
    except:
        return np.inf        
    
    #error propagation matrix
    M=np.zeros((3,Nb*3))
    for i in range(3):
        for j in range(Nb):
            for k in range(3):
                M[i,j*3+k]=A_plus[i,j]*A[j,k]
    
    var_U=np.sum(M[0,:]**2)
    var_V=np.sum(M[1,:]**2)
    
    var_u=0.5*(var_U+var_V)
    
    return var_u

def error_uu(alpha,beta):

    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([cosd(beta[i])*cosd(alpha[i]),cosd(beta[i])*sind(alpha[i]),sind(beta[i])])
        
    sa=sind(alpha)
    ca=cosd(alpha)
    sb=sind(beta)
    cb=cosd(beta)
    B=np.zeros((Nb,6))
    B[:,0]=cb**2*ca**2
    B[:,1]=cb**2*sa**2
    B[:,2]=sb**2
    B[:,3]=2*cb**2*ca*sa
    B[:,4]=2*cb*sb*ca
    B[:,5]=2*cb*sb*sa
    
    try:
        B_inv=np.linalg.inv(B)
    except:
        return np.inf      
    
    M=np.zeros((6,Nb*9))
    for i in range(6):
        for j in range(Nb):
            for k in range(3):
                for l in range(3):
                    M[i,j*9+k*3+l]=B_inv[i,j]*A[j,k]*A[j,l]
                    
    var_UU=np.sum(M[0,:]**2)
    var_VV=np.sum(M[1,:]**2)
    var_UV=np.sum(M[3,:]**2)
    
    var_uu=3/8*(var_UU+var_VV)+0.5*var_UV
    
    return var_uu


def error(theta_r,x0,y0,z0,weight):
    
    #extract scan geometry
    Nb=int(len(theta_r)/2)
    theta=theta_r[:Nb]
    r=theta_r[Nb:]
    
    azi,ele=pol2spher(theta,r,x0,y0,z0)
    assert len(azi)==len(ele), "Lenghts of azimuth and elevation mismatch"
    
    Nb=len(azi)
    beta=ele
    alpha=(90-azi)%360
    
    #calculate errors
    var_u=error_ws(alpha, beta)
    var_uu=error_uu(alpha, beta)
    
    return var_u*weight+var_uu*(1-weight)

#%% Initialization
turbines=xr.open_dataset(source_layout,group='turbines')

lidar_x=utm.from_latlon(lidar_lat,lidar_lon)[0]
lidar_y=utm.from_latlon(lidar_lat,lidar_lon)[1]

turb_x={}
turb_y={}
for t in turbine_sel:
    turb_x[t]=turbines.x_utm[turbines.name==t].values[0]-lidar_x
    turb_y[t]=turbines.y_utm[turbines.name==t].values[0]-lidar_y
    

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
            
            azi_opt.append(np.around(pol2spher(theta_opt,r_opt,x0,y0,z0)[0],2))
            ele_opt.append(np.around(pol2spher(theta_opt,r_opt,x0,y0,z0)[1],2))
            f_opt.append(np.around(res.fun,5))
        
            f_best.append(min(f_opt))
            azi_best.append(azi_opt[np.where(f_opt==f_best[-1])[0][0]])
            ele_best.append(ele_opt[np.where(f_opt==f_best[-1])[0][0]])
        else:
            print(res.message)
        print(t+': '+str(len(f_opt)))

    #plot
    for a,e in zip(azi_best[-1],ele_best[-1]):
        rh=z0/sind(e)
        plt.plot([0,rh*cosd(e)*cosd(90-a)],[0,rh*cosd(e)*sind(90-a)],[h_lidar,h_lidar+rh*sind(e)],color=colors[t])
    plt.plot(x0+cosd(np.arange(360))*rmax,y0+sind(np.arange(360))*rmax,np.arange(360)*0+z0,color=colors[t],linewidth=1)

    draw_turbine_3d(ax,turb_x[t],turb_y[t],turb_z[t]+h,d,h,180)
    
    #validation of WS error
    
    #cartesian->LOS matrix
    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([cosd(ele_best[-1][i])*cosd(90-azi_best[-1][i]),cosd(ele_best[-1][i])*sind(90-azi_best[-1][i]),sind(ele_best[-1][i])])
        
    i_wd=0
    ws_err=np.zeros(360)
    for wd in range(360):
        u0=cosd(270-wd)
        v0=sind(270-wd)
        w0=0
    
        vel_los=np.zeros((Nb,L))
        for i in range(L):
            vel_vector=np.zeros((3,Nb))
            vel_vector[0,:]=np.random.normal(0,0.1,Nb)+u0
            vel_vector[1,:]=np.random.normal(0,0.1,Nb)+v0
            vel_vector[2,:]=np.random.normal(0,0.1,Nb)+w0
            for j in range(Nb):
                vel_los[j,i]=A[j,:]@vel_vector[:,j]
        
        
        #reconstruct velocity
        A_plus=np.linalg.inv(A.T@A)@A.T
        vel_rec=A_plus@vel_los
        
        ws_rec=(vel_rec[0,:]**2+vel_rec[1,:]**2)**0.5
        
        
        ws_err[i_wd]=np.std(ws_rec-1)
        i_wd+=1
        print(wd)
    
    ws_err2=error_ws((90-azi_best[-1])%360,ele_best[-1])**0.5*0.1
    
    #validate error on TI
    
    sa=sind((90-azi_best[-1])%360)
    ca=cosd((90-azi_best[-1])%360)
    sb=sind(ele_best[-1])
    cb=cosd(ele_best[-1])
    B=np.zeros((Nb,6))
    B[:,0]=cb**2*ca**2
    B[:,1]=cb**2*sa**2
    B[:,2]=sb**2
    B[:,3]=2*cb**2*ca*sa
    B[:,4]=2*cb*sb*ca
    B[:,5]=2*cb*sb*sa
    
    B_inv=np.linalg.inv(B)
    
    i_wd=0
    
    RS0=np.array([2,0.5,0.3,0,-0.1,0])
    uu_err=np.zeros(360)
    for wd in range(360):
        
        M_rot=np.array([[sind(wd)**2,cosd(wd)**2,0, sind(2*wd),0,0],
                        [cosd(wd)**2,sind(wd)**2,0,-sind(2*wd),0,0],
                        [0,0,1,0,0,0],
                        [-0.5*sind(2*wd),0.5*sind(2*wd),0,-cosd(2*wd),0,0],
                        [0,0,0,0,-sind(wd),-cosd(wd)],
                        [0,0,0,0, cosd(wd),-sind(wd)]])
        
        RS=np.linalg.inv(M_rot)@RS0
        
        var_los=np.zeros((Nb,L))
        for i in range(L):
            RS_mc=np.zeros((6,Nb))
            RS_mc=np.tile(RS,(Nb,1)).T+np.random.normal(0,0.01,(6,Nb))
            for j in range(Nb):
                var_los[j,i]=B[j,:]@RS_mc[:,j]
        
        #reconstruct TI
        
        RS_rec=B_inv@var_los
        
        uu_rec=(RS_rec[0,:]*cosd(270-wd)**2+RS_rec[1,:]*sind(270-wd)**2+2*RS_rec[3,:]*cosd(270-wd)*sind(270-wd))

        uu_err[i_wd]=np.std(uu_rec-RS0[0])

        i_wd+=1
        print(wd)
    
    uu_err2=error_uu((90-azi_best[-1])%360,ele_best[-1])**0.5*0.01
    
    raise BaseException()
    
   
     
    
    
    
    
    

#%% Plots
ax.set_xlim([-750,750])
ax.set_ylim([-250,350])
ax.set_zlim([0,h+d])
ax.set_aspect('equal')
ax.set_xlabel('W-E [m]')
ax.set_ylabel('S-N [m]')
ax.set_zlabel(r'$z$ [m a.g.l.]')



