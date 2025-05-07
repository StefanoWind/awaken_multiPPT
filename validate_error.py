# -*- coding: utf-8 -*-
"""
Validate error formulas for scan design
"""

import os
cd=os.path.dirname(__file__)
import xarray as xr
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
azi=np.array([259.62, 277.17, 268.9 , 283.43, 267.35, 252.45])
ele=np.array([ 8.84,  8.91, 10.13, 11.85, 14.58, 11.57])

azi=np.array([213.33,  38.23,  70.7 ,  60.06, 217.52,  37.39])
ele=np.array([25.63, 12.28, 13.43, 13.36, 14.9 ,  6.29])

RS_rot=np.array([1,0.8,0.5,0,-0.1,0])#[IEC 61400-1]
err=0.01

wds=np.arange(360)
L=100

#%% Initialization
ws_err=np.zeros(len(wds))
U_err=np.zeros(len(wds))
V_err=np.zeros(len(wds))
ws_err2=np.zeros(len(wds))

uu_err=np.zeros(len(wds))
UU_err=np.zeros(len(wds))
VV_err=np.zeros(len(wds))
UV_err=np.zeros(len(wds))
uu_err2=np.zeros(len(wds))

Nb=len(azi)

#%% Main
#validation of WS error

#cartesian->LOS matrix
A=np.zeros((Nb,3))
for i in range(Nb):
    A[i,:]=np.array([cosd(ele[i])*cosd(90-azi[i]),cosd(ele[i])*sind(90-azi[i]),sind(ele[i])])
A_plus=np.linalg.inv(A.T@A)@A.T

i_wd=0
for wd in wds:
    u0=cosd(270-wd)
    v0=sind(270-wd)
    w0=0

    vel_los=np.zeros((Nb,L))
    for i in range(L):
        vel_vector=np.zeros((3,Nb))
        vel_vector[0,:]=np.random.normal(0,err,Nb)+u0
        vel_vector[1,:]=np.random.normal(0,err,Nb)+v0
        vel_vector[2,:]=np.random.normal(0,err,Nb)+w0
        for j in range(Nb):
            vel_los[j,i]=A[j,:]@vel_vector[:,j]
    
    #reconstruct velocity
    
    vel_rec=A_plus@vel_los
    ws_rec=(vel_rec[0,:]**2+vel_rec[1,:]**2)**0.5
    
    #store directional error on wind speed
    U_err[i_wd]=np.std(vel_rec[0,:]-u0)
    V_err[i_wd]=np.std(vel_rec[1,:]-v0)
    
    ws_err[i_wd]=np.std(ws_rec-1)
    ws_err2[i_wd]=error_ws((90-azi)%360,ele,wd)[0]*err
    i_wd+=1
    print(wd)

#omnidirectional error on wind speed
ws_err_avg=np.mean(ws_err)
ws_err2_avg,U_err2,V_err2=tuple(x*err for x in error_ws((90-azi)%360,ele))

#validate error on uu
sa=sind((90-azi)%360)
ca=cosd((90-azi)%360)
sb=sind(ele)
cb=cosd(ele)
B=np.zeros((Nb,6))
B[:,0]=cb**2*ca**2
B[:,1]=cb**2*sa**2
B[:,2]=sb**2
B[:,3]=2*cb**2*ca*sa
B[:,4]=2*cb*sb*ca
B[:,5]=2*cb*sb*sa

B_inv=np.linalg.inv(B)

i_wd=0
for wd in wds:
    M_rot=np.array([[sind(wd)**2,cosd(wd)**2,0, sind(2*wd),0,0],
                    [cosd(wd)**2,sind(wd)**2,0,-sind(2*wd),0,0],
                    [0,0,1,0,0,0],
                    [-0.5*sind(2*wd),0.5*sind(2*wd),0,-cosd(2*wd),0,0],
                    [0,0,0,0,-sind(wd),-cosd(wd)],
                    [0,0,0,0, cosd(wd),-sind(wd)]])
    
    RS0=np.linalg.inv(M_rot)@RS_rot
    
    var_los=np.zeros((Nb,L))
    for i in range(L):
        RS=np.zeros((6,Nb))
        RS=np.tile(RS0,(Nb,1)).T+np.random.normal(0,err,(6,Nb))
        for j in range(Nb):
            var_los[j,i]=B[j,:]@RS[:,j]
    
    #reconstruct uu
    RS_rec=B_inv@var_los
    uu_rec=RS_rec[0,:]*cosd(270-wd)**2+RS_rec[1,:]*sind(270-wd)**2+2*RS_rec[3,:]*cosd(270-wd)*sind(270-wd)
    
    #store errors on RS
    UU_err[i_wd]=np.std(RS_rec[0,:]-RS0[0])
    VV_err[i_wd]=np.std(RS_rec[1,:]-RS0[1])
    UV_err[i_wd]=np.std(RS_rec[3,:]-RS0[3])
    
    uu_err[i_wd]=np.std(uu_rec-RS_rot[0])
    uu_err2[i_wd]=error_uu((90-azi)%360,ele,wd)[0]*err

    i_wd+=1
    print(wd)

uu_err_avg=np.mean(uu_err)
uu_err2_avg,UU_err2,VV_err2,UV_err2=tuple(x*err for x in error_uu((90-azi)%360,ele))


#%% Plots
plt.close('all')
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.plot(wds,U_err,'k')
plt.plot(wds,U_err2*wds**0,'r')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{U})$')
plt.grid()
plt.ylim([0,np.max(ws_err)])

plt.subplot(1,3,2)
plt.plot(wds,V_err,'k')
plt.plot(wds,V_err2*wds**0,'r')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{V})$')
plt.grid()
plt.ylim([0,np.max(ws_err)])

plt.subplot(1,3,3)
plt.plot(wds,ws_err,'k',label=f'MC (mean={str(np.round(ws_err_avg,3))})')
plt.plot(wds,ws_err2,'r',label=f'Theory (mean={str(np.round(ws_err2_avg,3))})')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{u})$')
plt.grid()
plt.ylim([0,np.max(ws_err)])
plt.legend()
plt.tight_layout()

plt.figure(figsize=(18,4))
plt.subplot(1,4,1)
plt.plot(wds,UU_err,'k')
plt.plot(wds,UU_err2*wds**0,'r')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{U^\prime U^\prime})$')
plt.grid()
plt.ylim([0,np.max(uu_err)])

plt.subplot(1,4,2)
plt.plot(wds,VV_err,'k')
plt.plot(wds,VV_err2*wds**0,'r')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{V^\prime V^\prime})$')
plt.grid()
plt.ylim([0,np.max(uu_err)])

plt.subplot(1,4,3)
plt.plot(wds,UV_err,'k')
plt.plot(wds,UV_err2*wds**0,'r')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{U^\prime V^\prime})$')
plt.grid()
plt.ylim([0,np.max(uu_err)])

plt.subplot(1,4,4)
plt.plot(wds,uu_err,'k',label=f'MC (mean={str(np.round(uu_err_avg,3))})')
plt.plot(wds,uu_err2,'r',label=f'Theory (mean={str(np.round(uu_err2_avg,3))})')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\Delta^2(\overline{u^\prime u^\prime})$')
plt.grid()
plt.legend()
plt.ylim([0,np.max(uu_err)])
plt.tight_layout()
