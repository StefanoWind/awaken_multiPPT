# -*- coding: utf-8 -*-
"""
Utilities for scan design
"""


import os
cd=os.path.dirname(__file__)
import numpy as np

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))


def pol2spher(theta,r,x0,y0,z0):
    
    x=x0+r*cosd(theta)
    y=y0+r*sind(theta)
    z=z0
    
    alpha=np.degrees(np.arctan2(y,x))
    beta=np.degrees(np.arcsin(z/(x**2+y**2+z**2)**0.5))
    
    return (90-alpha)%360,beta


def draw_turbine_3d(ax,x,y,z,D,H,yaw):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Correct import
    from stl import mesh  # Correct import for Mesh
    
    # Load the STL file of the 3D turbine model
    turbine_mesh = mesh.Mesh.from_file(os.path.join(cd,'figures','blades.stl'))
    tower_mesh = mesh.Mesh.from_file(os.path.join(cd,'figures','tower.stl'))
    nacelle_mesh = mesh.Mesh.from_file(os.path.join(cd,'figures','nacelle.stl'))

    #translate
    translation_vector = np.array([-125, -110, -40])
    turbine_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -95, -150])
    tower_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -100,-10])
    nacelle_mesh.vectors += translation_vector

    #rescale
    scaling_factor = 1/175*D
    turbine_mesh.vectors *= scaling_factor

    scaling_factor = 1/250*D
    scaling_factor_z=1/0.6*H/D
    tower_mesh.vectors *= scaling_factor
    tower_mesh.vectors[:, :, 2] *= scaling_factor_z

    scaling_factor = 1/175*D
    nacelle_mesh.vectors *= scaling_factor

    #rotate
    theta = np.radians(180+yaw)  
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,             1]
    ])

    turbine_mesh.vectors = np.dot(turbine_mesh.vectors, rotation_matrix)
    tower_mesh.vectors = np.dot(tower_mesh.vectors, rotation_matrix)
    nacelle_mesh.vectors = np.dot(nacelle_mesh.vectors, rotation_matrix)

    #translate
    translation_vector = np.array([x, y, z])
    turbine_mesh.vectors += translation_vector
    tower_mesh.vectors += translation_vector
    nacelle_mesh.vectors += translation_vector


    # Extract the vertices from the rotated STL mesh
    faces = turbine_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Get the scale from the mesh to fit it properly
    scale = np.concatenate([turbine_mesh.points.min(axis=0), turbine_mesh.points.max(axis=0)])

    # Extract the vertices from the rotated STL mesh
    faces = tower_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Extract the vertices from the rotated STL mesh
    faces = nacelle_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)


    # Set the scale for the axis
    ax.auto_scale_xyz(scale, scale, scale)


def error_ws(alpha,beta,wd=None):
    Nb=len(alpha)
    
    #cartesian->LOS matrix
    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([cosd(beta[i])*cosd(alpha[i]),cosd(beta[i])*sind(alpha[i]),sind(beta[i])])

    #LOS->cartesian matrix
    try:
        A_plus=np.linalg.inv(A.T@A)@A.T
    except:
        return np.inf,np.inf,np.inf      
    
    #expanded solution matrix
    M=np.zeros((3,Nb*3))
    for i in range(3):
        for j in range(Nb):
            for k in range(3):
                M[i,j*3+k]=A_plus[i,j]*A[j,k]
    
    #error covariance
    S=M@M.T
    
    #error propagation
    if wd==None:
        J_Jt_avg=np.array([[0.5,0,0],[0,0.5,0],[0,0,0]])
        var_u=np.sum(J_Jt_avg*S)
    else:
        J=np.array([[cosd(270-wd)],[sind(270-wd)],[0]])
        var_u=np.sum(J@J.T*S)
    
    return var_u**0.5,S[0,0]**0.5,S[1,1]**0.5

def error_uu(alpha,beta,wd=None):
    Nb=len(alpha)
    
    #RS to LOS variance matrix
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
        return np.inf,np.inf,np.inf,np.inf          
    
    #expanded solution matrix
    M=np.zeros((6,Nb*6))
    for i in range(6):
        for j in range(Nb):
            for k in range(6):
                M[i,j*6+k]=B_inv[i,j]*B[j,k]
                    
    #error covariance
    S=M@M.T
    
    #error propagation
    if wd==None:
        J=np.array([[3/8],[3/8],[0],[0.5],[0],[0]])
    else:
        J=np.array([[cosd(270-wd)**2],[sind(270-wd)**2],[0],[2*cosd(270-wd)*sind(270-wd)],[0],[0]])
        
    var_uu=np.sum(J@J.T*S)
    return var_uu**0.5,S[0,0]**0.5,S[1,1]**0.5,S[3,3]**0.5

