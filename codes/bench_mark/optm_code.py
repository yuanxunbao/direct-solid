#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this code is optimized version of dirsolid_cuda_dp.py
Optimization ways:
1. [done] single precision
2. shared memory/tiling/coelsing
3. [done] optimize kernel calls/ threads per block
4. [done] optimize register by tmp variable

@author: yigong qin, yuanxun bao
"""
import importlib
import sys
import os
import scipy.io as sio
from scipy.io import savemat as save
from numba import njit, cuda, vectorize, float32, float64, int32
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numpy.random import random
import time
import math
from QoIs import *

PARA = importlib.import_module(sys.argv[1])
# import dsinput as PARA

'''
-------------------------------------------------------------------------------------------------
LOAD PARAMETERS
-------------------------------------------------------------------------------------------------
'''
if len(sys.argv) ==4:
      delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m_slope, G, R, Ti, U_0 = PARA.phys_para(sys.argv[2],int(sys.argv[3]))
elif len(sys.argv) ==3:
      delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m_slope, G, R, Ti, U_0 = PARA.phys_para(sys.argv[2])
else: delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m_slope, G, R, Ti, U_0 = PARA.phys_para()
eps, alpha0, lx, aratio, nx, dt, Mt, eta, \
seed_val, nts,direc, mvf, tip_thres, ictype, qts,qoi_winds = PARA.simu_para(W0,Dl_tilde,tau0)

# dimensionalize
lxd = lx * W0

mph = 'cell' if eta ==0.0 else 'dendrite'

filename = 'dirsolid'+ '_G'+str('%4.2F'%G) + '_R'+str('%4.2F'%R) + '_noise'+ \
str('%4.2F'%eta)+'_misori'+str(alpha0)+'_lx'+ str('%4.2F'%lxd)+'_nx'+str('%d'%nx)+'_asp'+str(aratio)+ \
'_ictype'+ str('%d'%ictype) + '_U0'+str('%4.2F'%U_0) + '_seed'+str(seed_val)

# calculate snapshot / qoi to save
kts = int( 2*np.floor((Mt/nts)/2) ); # print(kts)
nts = int(Mt/kts); print(nts)
interq = int( 2*np.floor(Mt/qts/2) ); # print(interq)
qts = int(Mt/interq); print(qts)

'''
-------------------------------------------------------------------------------------------------
CAST PARAMETERS INTO FLOAT32
-------------------------------------------------------------------------------------------------
'''

delta = float64(delta)
k = float64(k)
lamd = float64(lamd)
R_tilde = float64(R_tilde)
Dl_tilde = float64(Dl_tilde)
lT_tilde = float64(lT_tilde)
W0 = float64(W0)
tau0 = float64(tau0)

eps = float64(eps)
alpha0_rad = float64(alpha0*math.pi/180)
lx = float64(lx)
nx = int32(nx)
dt = float64(dt)
Mt = int32(Mt)
eta = float64(eta)
U_0 = float64(U_0)

cosa = float64(np.cos(alpha0_rad))
sina = float64(np.sin(alpha0_rad))

a_s = float64(1 - 3*delta)
epsilon = float64(4.0*delta/a_s)
a_12 = float64(4.0*a_s*epsilon)


sqrt2 = float64(np.sqrt(2.0))


lz = float64(aratio*lx)
nz = int32(aratio*nx+1)
nv= nz*nx #number of variables
dx = float64( lx/nx )
dz = float64( lz/(nz-1) ) 

x_1d = (np.linspace(0,lx-dx,nx)).astype(np.float64)
z_1d = (np.linspace(0,lz,nz)).astype(np.float64)
zz,xx = np.meshgrid(z_1d, x_1d)
print('mesh data type', zz.dtype )

dxdz_in = float64( 1./(dx*dz) ) 
dxdz_in_sqrt = float64( np.sqrt(dxdz_in) )

hi= float64( 1./dx )

dt_sqrt =  float64( np.sqrt(dt) )

dxd = float64( dx*W0 )

print('==========================================\n')
print('W0 = %4.2E um'% W0)
print('tau0 = %4.12E s'%tau0)
print('dx = %4.2E um'%(dx*W0))
print('dt = %4.12E s'%(dt*tau0))
print('lambda = ', lamd)
print('Mt = ', Mt)
print('U0 = ', U_0)
print('grid = %d x %d'%(nx,nz))
print('==========================================\n')

'''
-------------------------------------------------------------------------------------------------
ALLOCATE SPACE FOR OUTPUT ARRAYS
-------------------------------------------------------------------------------------------------
'''
op_phi = np.zeros((nv,nts+1), dtype=np.float64)
conc = np.zeros((nv,nts+1), dtype=np.float64)
zz_mv = np.zeros((nz,nts+1), dtype=np.float64)
t_snapshot = np.zeros(nts+1, dtype=np.float64)

'''
-------------------------------------------------------------------------------------------------
INIT. TIP POSITION
-------------------------------------------------------------------------------------------------
'''
cur_tip = 0	# global variable that stores the tip z-index

sum_arr = np.array([0], dtype=np.float64) # init sum = 0
Ntip = 1

@cuda.jit
def sum_cur_tip(d_sum_arr, d_tip, d_phi):

    m,n = d_phi.shape
    i = cuda.grid(1)
    val = 0. 
 
    if (0 < i < m-1):
        val = d_phi[i, d_tip]
    
    cuda.atomic.add(d_sum_arr, 0, val)


def compute_tip_pos(cur_tip,sum_arr, phi):

    while True :

        sum_arr[0] = 0.
        sum_cur_tip[bpg,tpb](sum_arr,  cur_tip, phi)
        mean_along_z = sum_arr[0] / nx

        if (mean_along_z > -0.99):
            cur_tip += 1
        else: 
            tip_x = np.argmax(phi[:,cur_tip])
            return tip_x, cur_tip

@njit
def set_halo(u):
    
    m,n = u.shape
    ub = np.zeros((m+2,n+2))
    
    ub[1:-1,1:-1] = u
    
    return ub



'''
Device function
'''
@cuda.jit('float64(float64, float64)',device=True)
def atheta(ux, uz):

    ux2 = (  cosa*ux + sina*uz )**2
    uz2 = ( -sina*ux + cosa*uz )**2
        
    # return MAG_sq2
    MAG_sq = (ux2 + uz2)
    MAG_sq2= MAG_sq**2
    
    if (MAG_sq > eps):
        
        return a_s*( 1 + epsilon*(ux2**2 + uz2**2) / MAG_sq2   )
        # return uz/MAG_sq2
    else:
        return 1.0 # a_s
    
    
@cuda.jit('float64(float64, float64)',device=True)
def aptheta(ux, uz):
    uxr =  cosa*ux + sina*uz
    uzr = -sina*ux + cosa*uz
    ux2 = uxr**2
    uz2 = uzr**2
    
    MAG_sq  = ux2 + uz2
    MAG_sq2 = MAG_sq**2
    
    if (MAG_sq > eps):
        
        return -a_12*uxr*uzr*(ux2 - uz2) /  MAG_sq2
    
    else:
        return 0.0
    
    
# set boundary conditions cpu version
def setBC_cpu(u,BCx,BCy):
    
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
def rhs_psi(ps, ph, U, ps_new, ph_new, U_new, zz, dpsi, nt, rng_states):
    # ps = psi, ph = phi

    # ==================================================
    #
    # 0. SHARED MEMORY LOAD
    #
    # ==================================================
    # 2D: ps, ph, U (no halo needed); 1D: zz (no halo needed) 
    

    i,j = cuda.grid(2)
    m,n = ps.shape

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

        # print(lT_tilde)         
        Up = (zz[j] - R_tilde * (nt*dt) )/lT_tilde
        # Up = (zz[i,j]-z0 - R_tilde * (nt*dt) )/lT_tilde

        rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
                   sqrt2*ph[i,j] - lamd*(1-ph[i,j]**2)*sqrt2*(U[i,j] + Up)


        # =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================
        tp = (1-(1-k)*Up)
        # tp = 1+(1-k)*U[i,j]
        tau_psi = tp*A2 if tp >= k else k*A2
        # tau_psi = A2        
        
        dpsi[i,j] = rhs_psi / tau_psi # + eta*(random()-0.5)/dt_sr
        
        #x = xoroshiro128p_uniform_float64(rng_states, thread_id)
        threadID = j*m + i
        beta_ij = xoroshiro128p_uniform_float64(rng_states, threadID) - 0.5 # rand from  [-0.5, 0.5]
        
        # update psi and phi
        ps_new[i,j] = ps[i,j] + ( dt * dpsi[i,j] + dt_sqrt*dxdz_in_sqrt*eta * beta_ij ) 
        ph_new[i,j] = math.tanh(ps_new[i,j]/sqrt2)



@cuda.jit
def moveframe(ps, ph, U, zz, ps_buff, ph_buff, U_buff, zz_buff):

    i,j = cuda.grid(2)
    m,n = ps.shape

    if i == 0 and 0 < j < n-2 :
        zz_buff[j] = zz[j+1]
        
    if i == 0 and j == n-2 :
        zz_buff[j] = 2*zz[n-2] - zz[n-3]

    if 0 < i < m-1 and 0 < j < n-2 :
        
        ps_buff[i,j] = ps[i,j+1]
        ph_buff[i,j] = ph[i,j+1]
        U_buff[i,j]  = U[i,j+1]
        # zz_buff[j] = zz[j+1]

    if 0 < i < m-1 and j == n-2 :
        ps_buff[i,j] = 2*ps[i,n-2] - ps[i,n-3] # extrapalation
        ph_buff[i,j] = 2*ph[i,n-2] - ph[i,n-3]
        U_buff[i,j]  = U_0 
        # zz_buff[j] = 2*zz[n-2] - zz[n-3]

@cuda.jit
def copyframe(ps_buff, ph_buff, U_buff, zz_buff, ps, ph, U, zz):

    i,j = cuda.grid(2)
    m,n = ps.shape

    if i == 0 and 0 < j < n-1 :
        zz[j] = zz_buff[j]
 
    if 0 < i < m-1 and 0 < j < n-1 :
        ps[i,j] = ps_buff[i,j]
        ph[i,j] = ph_buff[i,j]
        U[i,j]  = U_buff[i,j]
        # zz[j] = zz_buff[j]


@cuda.jit
def setBC_gpu(ps,ph,U,dpsi):
    
    m,n = ps.shape
    i = cuda.grid(1)
   
    # periodic at x = 0, lx
    if ( i < n ):
        ps[0,i]   = ps[m-2,i]
        ph[0,i]   = ph[m-2,i]
        U[0,i]    = U[m-2,i]
        dpsi[0,i] = dpsi[m-2,i]

        ps[m-1,i]   = ps[1,i]
        ph[m-1,i]   = ph[1,i]
        U[m-1,i]    = U[1,i]
        dpsi[m-1,i] = dpsi[1,i]


    # no-flux at z = 0, lz
    if ( i < m ):
        ps[i,0] = ps[i,2]
        ph[i,0] = ph[i,2]
        U[i,0] = U[i,2]
        dpsi[i,0] = dpsi[i,2]

        ps[i,n-1] = ps[i,n-3]
        ph[i,n-1] = ph[i,n-3]
        U[i,n-1] = U[i,n-3]
        dpsi[i,n-1] = dpsi[i,n-3]


@cuda.jit
def rhs_U(U, U_new, ph, dpsi):


    i,j = cuda.grid(2)
    m,n = U.shape

    if 0 < i < m-1 and 0 < j < n-1 :
 
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
        
        jat    = 0.5*(1+(1-k)*U[i,j])*(1-ph[i,j]**2)*dpsi[i,j]
        jat_ip = 0.5*(1+(1-k)*U[i+1,j])*(1-ph[i+1,j]**2)*dpsi[i+1,j]
    	
        UR = hi*Dl_tilde*0.5*(2 - ph[i,j] - ph[i+1,j])*(U[i+1,j]-U[i,j]) + \
    	     0.5*(jat + jat_ip)*nx
    	 
    	 
        # ============================
        # left edge flux (i-1/2, j)
        # ============================
        phx = ph[i+0,j+0]-ph[i-1,j+0]
        phz = phimjp - phimjm
        phn2 = phx**2 + phz**2
        nx = phx / math.sqrt(phn2) if (phn2 > eps) else 0.
        
        jat_im = 0.5*(1+(1-k)*U[i-1,j+0])*(1-ph[i-1,j+0]**2)*dpsi[i-1,j+0]
        
        UL = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i-1,j+0])*(U[i+0,j+0]-U[i-1,j+0]) + \
    	     0.5*(jat + jat_im)*nx
    	 
    	 
        # ============================
        # top edge flux (i, j+1/2)
        # ============================     
        phx = phipjp - phimjp
        phz = ph[i+0,j+1]-ph[i+0,j+0]
        phn2 = phx**2 + phz**2
        nz = phz / math.sqrt(phn2) if (phn2 > eps) else 0.
    	  
        jat_jp = 0.5*(1+(1-k)*U[i+0,j+1])*(1-ph[i+0,j+1]**2)*dpsi[i+0,j+1]      
        
        UT = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i+0,j+1])*(U[i+0,j+1]-U[i+0,j+0]) + \
    	     0.5*(jat + jat_jp)*nz
    	 
    	 
        # ============================
        # top edge flux (i, j-1/2)
        # ============================  
        phx = phipjm - phimjm
        phz = ph[i+0,j+0]-ph[i+0,j-1]
        phn2 = phx**2 + phz**2
        nz = phz / math.sqrt(phn2) if (phn2 > eps) else 0.
        
        jat_jm = 0.5*(1+(1-k)*U[i+0,j-1])*(1-ph[i+0,j-1]**2)*dpsi[i+0,j-1]      
        
        UB = hi*Dl_tilde*0.5*(2 - ph[i+0,j+0] - ph[i+0,j-1])*(U[i+0,j+0]-U[i+0,j-1]) + \
    	     0.5*(jat + jat_jm)*nz
        
        rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat
        tau_U = (1+k) - (1-k)*ph[i+0,j+0]

        U_new[i,j] = U[i,j] + dt * ( rhs_U / tau_U )




def save_data(phi,U,z):
    
    cinf_cl0 =  1+ (1-k)*U_0
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 ) / cinf_cl0
    
#    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )
   
    return np.reshape(phi[1:-1,1:-1],     (nv,1), order='F') , \
           np.reshape(c_tilde[1:-1,1:-1], (nv,1), order='F') , \
           z[1:-1,].T

def save_data_transient(psi,phi,U,z):

    cinf_cl0 =  1+ (1-k)*U_0
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 ) / cinf_cl0

#    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )

 
    return psi[1,1:-1],phi[1,1:-1],U[1,1:-1],c_tilde[1,1:-1],z[1:-1] 

if ictype == 0: 

    psi0 = PARA.seed_initial(xx,lx,zz)
    U0 = 0*psi0 + U_0
    phi0 = np.tanh(psi0/sqrt2)

elif ictype == 1:
  
    z0 = lz*0.0
    psi0 = PARA.planar_initial(lz,zz,z0)
    phi0 = np.tanh(psi0/sqrt2)

    U0 = 0*psi0 + U_0 


elif ictype == 2:

    z0 = lz * 0.0
    psi0 = PARA.sum_sine_initial(lx,nx,xx,zz,z0)
    phi0 = np.tanh(psi0/sqrt2)

    c_liq = c_inf + c_inf*(1.0-k)/k*np.exp(-R_tilde/Dl_tilde *(zz - z0 )) * (zz >= z0) 
    c_sol = c_inf

    U0 = 0*psi0 + U_0
#    U0 = 0 * (phi0 >= 0.0 ) + \
#         (k*c_liq/c_infty-1)/(1-k) * (phi0 < 0.0)

elif ictype == 3:

    z0 = lz*0.0
    psi0 = PARA.planar_initial(lz,zz,z0)
    phi0 = np.tanh(psi0/sqrt2)
    
    c_liq = c_inf + c_inf*(1.0-k)/k*np.exp(-R_tilde/Dl_tilde *(zz - z0 )) * (zz >= z0) 
    c_sol = c_inf

    U0 = 0 * (phi0 >= 0.0 ) + \
         (k*c_liq/c_inf-1)/(1-k) * (phi0 < 0.0)

elif ictype == 4:

    dd = sio.loadmat('initial2.mat')
    psi0 = dd['psi_ic']
    phi0 = dd['phi_ic']
    U0   = dd['U_ic']
    
else: 
    print('ERROR: invalid type of initial condtion...' )
    sys.exit(1)


# append halo around data
psi = set_halo(psi0)
phi = set_halo(phi0)
U = set_halo(U0)
zz = set_halo(zz)
z_cpu  = zz[1,:]

# set BCs
setBC_cpu(psi, 0, 1)
setBC_cpu(U, 0, 1)
setBC_cpu(phi,0,1)


phi_cpu = phi.astype(np.float64)
U_cpu = U.astype(np.float64)
psi_cpu = psi.astype(np.float64)


# save initial data
op_phi[:,[0]], conc[:,[0]], zz_mv[:,0] = save_data(phi_cpu, U_cpu, z_cpu )

# allocate space on device
psi_old = cuda.to_device(psi_cpu)
phi_old = cuda.to_device(phi_cpu)
U_old   = cuda.to_device(U_cpu)

print('c contiguous',psi_old.is_c_contiguous()); print('f contiguous',psi_old.is_f_contiguous());
psi_new = cuda.device_array_like(psi_old)
phi_new = cuda.device_array_like(phi_old)
U_new = cuda.device_array_like(U_old)

dPSI = cuda.device_array(psi_cpu.shape, dtype=np.float64)
z_gpu  = cuda.to_device(z_cpu)
z_gpu_buff = cuda.to_device(z_cpu)

# CUDA kernel invocation parameters
# cuda 2d grid parameters
tpb2d = (16,16)
bpg_x = math.ceil(phi.shape[0] / tpb2d[0])
bpg_y = math.ceil(phi.shape[1] / tpb2d[1])
bpg2d = (bpg_x, bpg_y)

# cuda 1d grid parameters 
tpb = 16 * 1
bpg = math.ceil( np.max( [phi.shape[0], phi.shape[1]] ) / tpb )

# CUDA random number states init
rng_states = create_xoroshiro128p_states( tpb2d[0]*tpb2d[1]*bpg_x*bpg_y, seed=seed_val)


print('2d threads per block: ({0:2d},{1:2d})'.format(tpb2d[0], tpb2d[1]))
print('2d blocks per grid: ({0:2d},{1:2d})'.format(bpg2d[0], bpg2d[1]))
print('(threads per block, block per grid) = ({0:2d},{1:2d})'.format(tpb, bpg))

# must be even
kts = int( 2*np.floor((Mt/nts)/2) ); # print(kts)
interq = int( 2*np.floor(Mt/qts/2) ); # print(interq)

start = time.time()

# march two steps per loop
for kt in range(int(Mt/2)):
   
    # =================================================================
    # time step: t = (2*nt) * dt
    # =================================================================
    rhs_psi[bpg2d, tpb2d](psi_old, phi_old, U_old, psi_new, phi_new, U_new, z_gpu, dPSI, 2*kt, rng_states)
#    setBC_gpu[bpg,tpb](psi_new, phi_new, U_old, dPSI)
#    rhs_U[bpg2d, tpb2d](U_old, U_new, phi_new, dPSI)

    # =================================================================
    # time step: t = (2*nt+1) * dt
    # ================================================================= 
    rhs_psi[bpg2d, tpb2d](psi_new, phi_new, U_new, psi_old, phi_old, U_old, z_gpu, dPSI, 2*kt+1, rng_states)
#    setBC_gpu[bpg,tpb](psi_old, phi_old, U_new, dPSI)
#    rhs_U[bpg2d, tpb2d](U_new, U_old, phi_old, dPSI) 
   
    # =================================================================
    # If moving frame flag is set to TRUE
    # =================================================================
    if mvf == True :
        # update tip position    
       # prev_tip = cur_tip
       cur_tip_x, cur_tip= compute_tip_pos(cur_tip, sum_arr, phi_old)    
       
       # when tip hit tip_thres, shift down by 1
       while cur_tip >= tip_thres:

           # new device arrays are used as buffer here, 
           moveframe[ bpg2d, tpb2d ](psi_old, phi_old, U_old, z_gpu, psi_new, phi_new, U_new, z_gpu_buff)
           copyframe[ bpg2d, tpb2d ](psi_new, phi_new, U_new, z_gpu_buff, psi_old, phi_old, U_old, z_gpu)
        
           # once frame is moved, BC needs to be updated again
           setBC_gpu[bpg,tpb]( psi_old, phi_old, U_old, dPSI  ) 
           
           # prev_tip = prev_tip-1
           cur_tip = cur_tip-1
    cur_tip_x, cur_tip= compute_tip_pos(cur_tip, sum_arr, phi_old)



end = time.time()
print('elapsed time: ', (end-start))

    # print & save data
if (2*kt+2)%kts==0:

       print('time step = ', 2*(kt+1) )
       if mvf == True: print('tip position nz = ', cur_tip)
       print('tip position nz = ', cur_tip)
       print('estimated increase of tip position if R is reached', (Mt/nts)*dt*tau0*R/(dx*W0))
       kk = int(np.floor((2*kt+2)/kts))
       phi = phi_old.copy_to_host()
       U  = U_old.copy_to_host()
       z_cpu = z_gpu.copy_to_host()
       op_phi[:,[kk]], conc[:,[kk]], zz_mv[:,kk] = save_data(phi,U,z_cpu)

       t_snapshot[kk] = 2*(kt+1)*dt

save(os.path.join(direc,filename+'.mat'),{'op_phi':op_phi, 'conc':conc, 'xx':xx*W0, 'zz_mv':zz_mv*W0,'dt':dt*tau0,\
 'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start, 't_snapshot':t_snapshot*tau0} )

