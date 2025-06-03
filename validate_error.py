# -*- coding: utf-8 -*-
"""
Validate error formulas for scan design
"""

import os
cd=os.path.dirname(__file__)
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import utils as utl
import glob
import pandas as pd

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18


#%% Inputs
source=os.path.join(cd,'data','scans','*xlsx')
RS_rot=np.array([1,0.8,0.5,0,-0.1,0])#[IEC 61400-1]
err=0.01#error on flow field

wds=np.arange(360)#[deg] wind direction loop
L=1000#MC draws

#%% Initialization
files=glob.glob(source)

#%% Main

for f in files:
    scan=pd.read_excel(f)
    if os.path.isfile(f.replace('xlsx','png')):
        continue
    azi=scan['Azimuth'].values
    ele=scan['Elevation'].values
    Nb=len(azi)
    
    #zeroing
    u_var=np.zeros(len(wds))
    u_var2=np.zeros(len(wds))
    
    uu_var=np.zeros(len(wds))
    uu_var2=np.zeros(len(wds))
    
    #%% Main
    #validation of WS error
    
    #cartesian->LOS matrix
    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([utl.cosd(ele[i])*utl.cosd(90-azi[i]),utl.cosd(ele[i])*utl.sind(90-azi[i]),utl.sind(ele[i])])
    A_plus=np.linalg.inv(A.T@A)@A.T
    
    i_wd=0
    for wd in wds:
        
        #nominal wind vector
        u0=utl.cosd(270-wd)
        v0=utl.sind(270-wd)
        w0=0
    
        #perturbed wind vectors
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
        u_rec=(vel_rec[0,:]**2+vel_rec[1,:]**2)**0.5
        
        #store directional error on wind speed
        u_var[i_wd]=np.mean((u_rec-1)**2)
        u_var2[i_wd]=utl.error_ws((90-azi)%360,ele,wd)*err**2
        i_wd+=1
        print(wd)
    
    #omnidirectional error on wind speed
    u_var_avg=np.mean(u_var)
    u_var2_avg=utl.error_ws((90-azi)%360,ele)*err**2
    
    #validate error on uu
    sa=utl.sind((90-azi)%360)
    ca=utl.cosd((90-azi)%360)
    sb=utl.sind(ele)
    cb=utl.cosd(ele)
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
        M_rot=np.array([[utl.sind(wd)**2,utl.cosd(wd)**2,0, utl.sind(2*wd),0,0],
                        [utl.cosd(wd)**2,utl.sind(wd)**2,0,-utl.sind(2*wd),0,0],
                        [0,0,1,0,0,0],
                        [-0.5*utl.sind(2*wd),0.5*utl.sind(2*wd),0,-utl.cosd(2*wd),0,0],
                        [0,0,0,0,-utl.sind(wd),-utl.cosd(wd)],
                        [0,0,0,0, utl.cosd(wd),-utl.sind(wd)]])
        
        RS0=np.linalg.inv(M_rot)@RS_rot
        
        var_los=np.zeros((Nb,L))
        for i in range(L):
            RS=np.zeros((6,Nb))
            RS=np.tile(RS0,(Nb,1)).T+np.random.normal(0,err,(6,Nb))
            for j in range(Nb):
                var_los[j,i]=B[j,:]@RS[:,j]
        
        #reconstruct uu
        RS_rec=B_inv@var_los
        uu_rec=RS_rec[0,:]*utl.cosd(270-wd)**2+RS_rec[1,:]*utl.sind(270-wd)**2+2*RS_rec[3,:]*utl.cosd(270-wd)*utl.sind(270-wd)
        
        #store errors on RS
        uu_var[i_wd]=np.mean((uu_rec-RS_rot[0])**2)
        uu_var2[i_wd]=utl.error_uu((90-azi)%360,ele,wd)*err**2
    
        i_wd+=1
        print(wd)
    
    uu_var_avg=np.mean(uu_var)
    uu_var2_avg=utl.error_uu((90-azi)%360,ele)*err**2
    
    #%% Plots
    plt.close('all')
    plt.figure(figsize=(18,6))
    
    plt.subplot(1,2,1)
    plt.plot(wds,u_var,'k',label=f'MC (mean={str(np.round(u_var_avg,4))})')
    plt.plot(wds,u_var2,'r',label=f'Theory (mean={str(np.round(u_var2_avg,4))})')
    plt.xlabel(r'$\theta_w$ [$^\circ$]')
    plt.ylabel(r'$\langle \Delta\hat{u}^2 \rangle$')
    plt.grid()
    plt.ylim([0,np.max(u_var)])
    plt.text(10,np.max(u_var)/10,str(os.path.basename(f)),bbox={'facecolor':'w','alpha':0.5,'edgecolor':'k'})
    
    plt.subplot(1,2,2)
    plt.plot(wds,uu_var,'k',label=f'MC (mean={str(np.round(uu_var_avg,4))})')
    plt.plot(wds,uu_var2,'r',label=f'Theory (mean={str(np.round(uu_var2_avg,4))})')
    plt.xlabel(r'$\theta_w$ [$^\circ$]')
    plt.ylabel(r'$\langle \Delta\hat{\overline{u^\prime}}^2 \rangle$')
    plt.grid()
    plt.legend()
    plt.ylim([0,np.max(uu_var)])
    plt.tight_layout()
    plt.savefig(f.replace('xlsx','png'))
    plt.close()
