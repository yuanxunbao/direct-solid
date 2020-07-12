#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:10:13 2020

@author: yigong qin, yuanxun bao
"""
import importlib
import sys
import os
from scipy.io import savemat as save
from numba import njit, stencil, vectorize, float32, float64, cuda
import numpy as np
import cupy as cp
from numpy.random import random
import time
import math
from math import pi
#PARA = importlib.import_module(sys.argv[1])
import dsinput as PARA

delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0 = PARA.phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename = PARA.simu_para(W0,Dl_tilde)
U_0, seed, nts, direc = PARA.IO_para(W0,lxd)

alpha0 = alpha0*pi/180

#eps = np.float32(eps)

#cosa = np.float32(np.cos(alpha0))
#sina = np.float32(np.sin(alpha0))

cosa = np.cos(alpha0)
sina = np.sin(alpha0)

a_s = np.float64(1 - 3*delta)
epsilon = np.float64(4*delta/a_s)
a_12 = np.float64(4*a_s*epsilon)


sqrt2 = np.sqrt(2.)

lx = lxd/W0

ratio = aratio
lz = ratio*lx

nz = int(ratio*nx+1)

nv= nz*nx #number of variables
dx = lx/nx
dz = lz/(nz-1)


x = np.linspace(0,lx-dx,nx)

z = np.linspace(0,lz,nz)

xx, zz = np.meshgrid(x, z)
t=0

dxdz_in = 1./(dx*dz)  

hi= 1./dx
dt_sr = np.sqrt(dt)

Tishot = np.zeros((2*nv,nts+1))

np.random.seed(seed)

@njit
def set_halo(u):
    
    m,n = u.shape
    ub = np.zeros((m+2,n+2))
    
    ub[1:-1,1:-1] = u
    
    return ub



'''
Device function
'''

#@cuda.jit('float32(float32, float32)',device=True)
@cuda.jit(device=True)
def atheta(ux, uz):

    ux2 = (  cosa*ux + sina*uz )**2
    uz2 = ( -sina*ux + cosa*uz )**2
        
    # return MAG_sq2
    MAG_sq2 = (ux2 + uz2)**2
    
    if (MAG_sq2 > eps**2):
        
        return a_s*( 1 + epsilon*(ux2**2 + uz2**2) / MAG_sq2   )
        # return uz/MAG_sq2
    else:
        return a_s
    
    
#@cuda.jit('float32(float32, float32)',device=True)
@cuda.jit(device=True)
def aptheta(ux, uz):
    uxr =  cosa*ux + sina*uz
    uzr = -sina*ux + cosa*uz
    ux2 = uxr**2
    uz2 = uzr**2
    
    MAG_sq2 = (ux2 + uz2)**2
    
    if (MAG_sq2 > eps**2):
        
        return -a_12*uxr*uzr*(ux2 - uz2) /  MAG_sq2
    
    else:
        return 0.0
    
    
# update global array, don't jit compile this function
def set_BC(u,BCx,BCy):
    
    # 0 periodic, 1 no flux (Neumann)
    
    if BCx == 0 :
                
        u[ 0,:] = u[-2,:] # 1st =  last - 1
        u[-1,:] = u[ 1,:] # last = second
        
    if BCx == 1 :
        
        u[ 0,:] = u[ 2,:] # 1st = 3rd
        u[-1,:] = u[-3,:] # last = last - 2
    
    if BCy == 0 :
        
        u[:, 0]  = u[:,-2]
        u[:,-1]  = u[:, 1]
        
    if BCy == 1 :
        
        u[:,0] = u[:,2]
        u[:,-1] = u[:,-3]
        
        
    return u


@cuda.jit
def atheta_gpu(px,py,out):
    
     x, y = cuda.grid(2)
     if 0 <  x < px.shape[0]-1 and 0 <  y < px.shape[1]-1 :
         for jj in range(1000):
             out[x,y] += atheta(out[x+1,y+1],py[x-1,y-1])
        

@cuda.jit
def sim_dirsolid(ps,ph,U,zz,ps_t):
    
    # ps = psi, ph = phi

    i,j = cuda.grid(2)
    m,n = ps.shape
    
    # time loop
    for nt in range(1):

        cuda.syncthreads()
        
        # thread on interior points
        if 0 < i < m-1 and 0 < j < n-1:

            # =============================================================
            #
            # 1. ANISOTROPIC DIFFUSION
            #
            # =============================================================
    
            # these ps's are defined on cell centers
            psipjp=( ps[i + 1, j + 1] + ps[i + 0, j + 1] + ps[i + 0, j + 0] + ps[i + 1, j + 0] ) * 0.25
            psipjm=( ps[i + 1, j + 0] + ps[i + 0, j + 0] + ps[i + 0, j - 1] + ps[i + 1, j - 1] ) * 0.25
            psimjp=( ps[i + 0, j + 1] + ps[i - 1, j + 1] + ps[i - 1, j + 0] + ps[i + 0, j + 0] ) * 0.25
            psimjm=( ps[i + 0, j + 0] + ps[i - 1, j + 0] + ps[i - 1, j - 1] + ps[i + 0, j - 1] ) * 0.25
    
            phipjp=( ph[i + 1, j + 1] + ph[i + 0, j + 1] + ph[i + 0, j + 0] + ph[i + 1, j + 0] ) * 0.25
            phipjm=( ph[i + 1, j + 0] + ph[i + 0, j + 0] + ph[i + 0, j - 1] + ph[i + 1, j - 1] ) * 0.25
            phimjp=( ph[i + 0, j + 1] + ph[i - 1, j + 1] + ph[i - 1, j + 0] + ph[i + 0, j + 0] ) * 0.25
            phimjm=( ph[i + 0, j + 0] + ph[i - 1, j + 0] + ph[i - 1, j - 1] + ph[i + 0, j - 1] ) * 0.25 
            
            # ============================
            # right edge flux
            # ============================
            psx = ps[i+1,j+0]-ps[i+0,j+0]
            psz = psipjp - psipjm
            phx = ph[i+1,j+0]-ph[i+0,j+0]
            phz = phipjp - phipjm
    
            A  = atheta( phx,phz)
            Ap = aptheta(phx,phz)
            JR = A * ( A*psx - Ap*psz )
            
            # ============================
            # left edge flux
            # ============================
            psx = ps[i+0,j+0]-ps[i-1,j+0]
            psz = psimjp - psimjm
            phx = ph[i+0,j+0]-ph[i-1,j+0]
            phz = phimjp - phimjm
    
            A  = atheta( phx,phz)
            Ap = aptheta(phx,phz)
            JL = A * ( A*psx - Ap*psz )
            
            # ============================
            # top edge flux
            # ============================
            psx = psipjp - psimjp
            psz = ps[i+0,j+1]-ps[i+0,j+0]
            phx = phipjp - phimjp
            phz = ph[i+0,j+1]-ph[i+0,j+0]

            A  = atheta( phx,phz)
            Ap = aptheta(phx,phz)
            JT = A * ( A*psz + Ap*psx )
    
            # ============================
            # bottom edge flux
            # ============================
            psx = psipjm - psimjm
            psz = ps[i+0,j+0]-ps[i+0,j-1]
            phx = phipjm - phimjm
            phz = ph[i+0,j+0]-ph[i+0,j-1]
    
            A  = atheta( phx,phz)
            Ap = aptheta(phx,phz)
            JB = A * ( A*psz + Ap*psx )
            
            
            # =============================================================
            #
            # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
            #
            # =============================================================
    
            # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)
            phxn = ( ph[i + 1, j + 0] - ph[i - 1, j + 0] ) * 0.5
            phzn = ( ph[i + 0, j + 1] - ph[i + 0, j - 1] ) * 0.5
            psxn = ( ps[i + 1, j + 0] - ps[i - 1, j + 0] ) * 0.5
            pszn = ( ps[i + 0, j + 1] - ps[i + 0, j - 1] ) * 0.5
    
            A2 = atheta(phxn,phzn)**2
            gradps2 = (psxn)**2 + (pszn)**2
            extra =  -sqrt2 * A2 * ph[i,j] * gradps2
    
    
            # =============================================================
            #
            # 3. double well (transformed): sqrt2 * phi + nonlinear terms
            #
            # =============================================================
    
            tn = nt * dt
            Up = (zz[i,j] - R_tilde * tn )/lT_tilde
    
            rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
                       sqrt2*ph[i,j] - lamd*(1-ph[i,j]**2)*sqrt2*(U[i,j] + Up)
    
    
            # =============================================================
            #
            # 4. dpsi/dt term
            #
            # =============================================================
            tp = (1-(1-k)*Up)
            tau_psi = tp*A2 if tp >= k else k*A2
            
            ps_t[i,j] = rhs_psi / tau_psi # + eta*(random()-0.5)/dt_sr

            # ps[i,j] += dt * ps_t[i,j]
            # ph[i,j] = math.tanh( ps[i,j] / sqrt2 )


        cuda.syncthreads()

        
        # set BC of dpsi/dt, and ph
        # to be used when updating U
        if i == 0 :
            ps_t[0,j] = ps_t[m-2,j] # periodic
            #ph[0,j] = ph[n-2,j] # periodic
                 
        if i == m-1 :
            ps_t[m-1,j] = ps_t[1,j] # periodic 
                #ph[m-1,j] = ph[1,j] # periodic 

        if j == 0 :
            ps_t[i,0] = ps_t[i,2] # no-flux
                #ph[i,0] = ph[i,2] # no-flux
        
        if j == n-1 :
            ps_t[i,n-1] = ps_t[i, n-3] # no-flux
                #ph[i,n-1] = ph[i, n-3] # no-flux

        cuda.syncthreads()

        

        
        if 0 < i < m-1 and 0 < j < n-1 :
            # =============================================================
            #
            # 5. update U term
            #
            # =============================================================

              
            # define cell centered values
            phipjp=( ph[i + 1, j + 1] + ph[i + 0, j + 1] + ph[i + 0, j + 0] + ph[i + 1, j + 0] ) * 0.25
            phipjm=( ph[i + 1, j + 0] + ph[i + 0, j + 0] + ph[i + 0, j - 1] + ph[i + 1, j - 1] ) * 0.25
            phimjp=( ph[i + 0, j + 1] + ph[i - 1, j + 1] + ph[i - 1, j + 0] + ph[i + 0, j + 0] ) * 0.25
            phimjm=( ph[i + 0, j + 0] + ph[i - 1, j + 0] + ph[i - 1, j - 1] + ph[i + 0, j - 1] ) * 0.25 
        
            # ============================
            # right edge flux (i+1/2, j)
            # ============================
            phx = ph[i+1,j]-ph[i,j]
            phz = phipjp - phipjm
            phn2 = phx**2 + phz**2
            nx = phx / math.sqrt(phn2) if (phn2 > eps) else 0.
            
            jat    = 0.5*(1+(1-k)*U[i,j])*(1-ph[i,j]**2)*ps_t[i,j]
            jat_ip = 0.5*(1+(1-k)*U[i+1,j])*(1-ph[i+1,j]**2)*ps_t[i+1,j]
                
            UR = hi*Dl_tilde*0.5*(2 - ph[i,j] - ph[i+1,j])*(U[i+1,j]-U[i,j]) + \
                 0.5*(jat + jat_ip)*nx
                 
                 
            # ============================
            # left edge flux (i-1/2, j)
            # ============================
            phx = ph[i+0,j+0]-ph[i-1,j+0]
            phz = phimjp - phimjm
            phn2 = phx**2 + phz**2
            nx = phx / math.sqrt(phn2) if (phn2 > eps) else 0.
            
            jat_im = 0.5*(1+(1-k)*U[i-1,j+0])*(1-ph[i-1,j+0]**2)*ps_t[i-1,j+0]
            
            UL = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i-1,j+0])*(U[i+0,j+0]-U[i-1,j+0]) + \
                 0.5*(jat + jat_im)*nx
                 
                 
            # ============================
            # top edge flux (i, j+1/2)
            # ============================     
            phx = phipjp - phimjp
            phz = ph[i+0,j+1]-ph[i+0,j+0]
            phn2 = phx**2 + phz**2
            nz = phz / math.sqrt(phn2) if (phn2 > eps) else 0.
                  
            jat_jp = 0.5*(1+(1-k)*U[i+0,j+1])*(1-ph[i+0,j+1]**2)*ps_t[i+0,j+1]      
            
            UT = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i+0,j+1])*(U[i+0,j+1]-U[i+0,j+0]) + \
                 0.5*(jat + jat_jp)*nz
                 
                 
            # ============================
            # top edge flux (i, j-1/2)
            # ============================  
            phx = phipjm - phimjm
            phz = ph[i+0,j+0]-ph[i+0,j-1]
            phn2 = phx**2 + phz**2
            nz = phz / math.sqrt(phn2) if (phn2 > eps) else 0.
            
            jat_jm = 0.5*(1+(1-k)*U[i+0,j-1])*(1-ph[i+0,j-1]**2)*ps_t[i+0,j-1]      
            
            UB = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i+0,j-1])*(U[i+0,j+0]-U[i+0,j-1]) + \
                 0.5*(jat + jat_jm)*nz
            
            rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat
            tau_U = (1+k) - (1-k)*ph[i+0,j+0]
           
    # race condtion! bad        
            # cuda.atomic.add(U, (i,j), dt*( rhs_U/tau_U ))
            # cuda.atomic.add(ps, (i,j), dt*ps_t[i,j] )
            
            
           #  U[i,j]  += dt * ( rhs_U/tau_U )
           #  ps[i,j] += dt * ps_t[i,j]
            
        
        cuda.syncthreads()
        # update 
        if 0 < i < m-1 and 0 < j < n-1 :
            ph[i,j] = math.tanh( ps[i,j] / sqrt2 )
             
            U[i,j]  += dt * ( rhs_U/tau_U )
            ps[i,j] += dt * ps_t[i,j]
            
        cuda.syncthreads()

        # set BCs for U
        if i == 0 : 
            U[0,j] = U[m-2,j] # periodic 
            ps[0,j] = ps[m-2,j] # periodic
            ph[0,j] = ph[m-2,j] # periodic 
                
                
        if i == m-1 :
            U[m-1,j] = U[1,j] # periodic 
            ps[m-1,j] = ps[1,j] # periodic 
            ph[m-1,j] = ph[1,j] # periodic 

        if j == 0 :
            U[i,0] = U[i,2] 
            ps[i,0] = ps[i,2]
            ph[i,0] = ph[i,2]

        if j == n-1 :
            U[i,n-1] = U[i, n-3]  
            ps[i,n-1] = ps[i, n-3]
            ph[i,n-1] = ph[i, n-3]



'''
@stencil
def _rhs_psi(ps,ph,U,zz):

    # ps = psi, ph = phi
    
    # =============================================================
    # 
    # 1. ANISOTROPIC DIFFUSION
    # 
    # =============================================================
    
    # these ps's are defined on cell centers
    psipjp=( ps[ 1, 1] + ps[ 0, 1] + ps[ 0, 0] + ps[ 1, 0] ) * 0.25
    psipjm=( ps[ 1, 0] + ps[ 0, 0] + ps[ 0,-1] + ps[ 1,-1] ) * 0.25
    psimjp=( ps[ 0, 1] + ps[-1, 1] + ps[-1, 0] + ps[ 0, 0] ) * 0.25
    psimjm=( ps[ 0, 0] + ps[-1, 0] + ps[-1,-1] + ps[ 0,-1] ) * 0.25
    
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * 0.25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * 0.25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * 0.25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * 0.25
    
    # ============================
    # right edge flux
    # ============================
    psx = ps[1,0]-ps[0,0]
    psz = psipjp - psipjm
    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JR = A * ( A*psx - Ap*psz )
    
    # ============================
    # left edge flux
    # ============================
    psx = ps[0,0]-ps[-1,0]
    psz = psimjp - psimjm
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JL = A * ( A*psx - Ap*psz )
    
    # ============================
    # top edge flux
    # ============================
    psx = psipjp - psimjp
    psz = ps[0,1]-ps[0,0]
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]


    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JT = A * ( A*psz + Ap*psx )
    
    # ============================
    # bottom edge flux
    # ============================
    psx = psipjm - psimjm
    psz = ps[0,0]-ps[0,-1]
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JB = A * ( A*psz + Ap*psx )
    
    
    
    # =============================================================
    # 
    # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
    # 
    # =============================================================
    
    # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)
    phxn = ( ph[ 1, 0] - ph[-1, 0] ) * 0.5
    phzn = ( ph[ 0, 1] - ph[ 0,-1] ) * 0.5
    psxn = ( ps[ 1, 0] - ps[-1, 0] ) * 0.5
    pszn = ( ps[ 0, 1] - ps[ 0,-1] ) * 0.5
    
    A2 = atheta(phxn,phzn)**2
    gradps2 = (psxn)**2 + (pszn)**2
    extra =  -sqrt2 * A2 * ph[0,0] * gradps2
    

    # =============================================================
    # 
    # 3. double well (transformed): sqrt2 * phi + nonlinear terms
    # 
    # =============================================================
    
    Up = (zz[0,0] )/lT_tilde
    
    rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
               sqrt2*ph[0,0] - lamd*(1-ph[0,0]**2)*sqrt2*(U[0,0] + Up) 
        
    
    # =============================================================
    # 
    # 4. dpsi/dt term
    # 
    # =============================================================
    tp = (1-(1-k)*Up)
    tau_psi = tp*A2 if tp >= k else k*A2

    return rhs_psi/tau_psi + eta*(random()-0.5)/dt_sr


@stencil
def _rhs_U(U,ph,psi_t):
    
    # define cell centered values
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * 0.25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * 0.25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * 0.25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * 0.25

    # ============================
    # right edge flux (i+1/2, j)
    # ============================
    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    phn2 = phx**2 + phz**2
    nx = phx / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat    = 0.5*(1+(1-k)*U[0,0])*(1-ph[0,0]**2)*psi_t[0,0]
    jat_ip = 0.5*(1+(1-k)*U[1,0])*(1-ph[1,0]**2)*psi_t[1,0]
        
    UR = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[1,0])*(U[1,0]-U[0,0]) + \
         0.5*(jat + jat_ip)*nx
         
         
    # ============================
    # left edge flux (i-1/2, j)
    # ============================
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    phn2 = phx**2 + phz**2
    nx = phx / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_im = 0.5*(1+(1-k)*U[-1,0])*(1-ph[-1,0]**2)*psi_t[-1,0]
    
    UL = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[-1,0])*(U[0,0]-U[-1,0]) + \
         0.5*(jat + jat_im)*nx
         
         
    # ============================
    # top edge flux (i, j+1/2)
    # ============================     
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
          
    jat_jp = 0.5*(1+(1-k)*U[0,1])*(1-ph[0,1]**2)*psi_t[0,1]      
    
    UT = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[0,1])*(U[0,1]-U[0,0]) + \
         0.5*(jat + jat_jp)*nz
         
         
    # ============================
    # top edge flux (i, j-1/2)
    # ============================  
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_jm = 0.5*(1+(1-k)*U[0,-1])*(1-ph[0,-1]**2)*psi_t[0,-1]      
    
    UB = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[0,-1])*(U[0,0]-U[0,-1]) + \
         0.5*(jat + jat_jm)*nz
    
    rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat
    tau_U = (1+k) - (1-k)*ph[0,0]
    
    
    return rhs_U/tau_U


# @njit(parallel=True)
@njit
def rhs_psi(ps,ph,U,zz): return _rhs_psi(ps,ph,U,zz)


# @njit(parallel=True)
@njit
def rhs_U(U,ph,psi_t): return _rhs_U(U,ph,psi_t)

'''

def save_data(phi,U):
    
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )
    
    return np.vstack(( np.reshape(phi[1:-1,1:-1].T,(nv,1),order='F') , \
                      np.reshape(c_tilde[1:-1,1:-1].T,(nv,1),order='F') ) )




##############################################################################

psi0 = PARA.seed_initial(xx,lx,zz)
U0 = 0*psi0 + U_0

psi = set_halo(psi0.T)
U = set_halo(U0.T)
zz = set_halo(zz.T)

set_BC(psi, 0, 1)
phi = np.tanh(psi/sqrt2)   # expensive replace
set_BC(U, 0, 1)
set_BC(phi,0,1)

# Tishot[:,[0]] = save_data(phi,U)


#complie
# start = time.time()
# dPSI = rhs_psi(psi, phi, U, zz)
# dPSI = set_BC(dPSI, 0, 1)
# dU = rhs_U(U,phi,dPSI)

# end = time.time()

# print('elapsed: ', end - start )


# arrays on device
psi_cp = cp.asarray(psi, dtype=cp.float64)
phi_cp = cp.asarray(phi, dtype=cp.float64)
U_cp   = cp.asarray(U  , dtype=cp.float64)
zz_cp  = cp.asarray(zz , dtype=cp.float64)
dPSI_cp = cp.zeros(zz_cp.shape, dtype = cp.float64)



threadsperblock = (16,16)
blockspergrid_x = math.ceil(phi.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(phi.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
# iincrement_a_2D_array[blockspergrid, threadsperblock](an_array)

print('threadsperblock : ', threadsperblock) 
print('blockspergrid : ', blockspergrid)

start2 = time.time()
sim_dirsolid[ blockspergrid, threadsperblock ](psi_cp, phi_cp, U_cp, zz_cp, dPSI_cp)
end2=time.time()

print('elapse2: ', (end2-start2))

#for jj in range(nts):

#    for ii in range(int(Mt/nts)):

#       dPSI = rhs_psi(psi, phi, U, zz - R_tilde*t)
   
# rrhs_psi[blockspergrid, threadsperblock](psi_cp,phi_cp,out_cp)

# atheta_gpu[blockspergrid, threadsperblock](phi_cp,psi_cp, out_cp)
        

#         dPSI = set_BC(dPSI, 0, 1)
    
#         psi = psi + dt*dPSI 
      
#         U = U + dt*rhs_U(U,phi,dPSI)
        
        # add boundary
#         psi = set_BC(psi, 0, 1)
    
#         U = set_BC(U, 0, 1)
        
#         phi = np.tanh(psi/sqrt2) 
        
        
 #        t += dt

    #print('now time is ',t)  
#     Tishot[:,[jj+1]] = save_data(phi,U)

t1 = time.time()
dPSI_cuda = cp.asnumpy(dPSI_cp)
psi_cuda = cp.asnumpy(psi_cp)
U_cuda = cp.asnumpy(U_cp)
phi_cuda = cp.asnumpy(phi_cp)
t2 = time.time()

print('transfer time = ', t2-t1)

print((dPSI_cuda))
print(np.max(U_cuda))
print(np.min(psi_cuda))

save('cuda.mat',{'psi_cuda':psi_cuda,'U_cuda':U_cuda, 'dPSI_cuda':dPSI_cuda,'phi_cuda':phi_cuda})
# Uf = U[1:-1,1:-1].T

# save(os.path.join(direc,filename),{'xx':xx*W0,'zz':zz[1:-1,1:-1].T*W0,'y':Tishot,'dt':dt*tau0,'nx':nx,'nz':nz,'t':t*tau0,'mach_time':end-start})
