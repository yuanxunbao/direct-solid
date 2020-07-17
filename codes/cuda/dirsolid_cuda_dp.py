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
from numba import njit, cuda, vectorize, float64, float64, int32
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numpy.random import random
import time
import math

PARA = importlib.import_module(sys.argv[1])
# import dsinput as PARA


'''
-------------------------------------------------------------------------------------------------
LOAD PARAMETERS
-------------------------------------------------------------------------------------------------
'''
delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0 = PARA.phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, seed_val, filename = PARA.simu_para(W0,Dl_tilde)
U_0, seed, nts, direc = PARA.IO_para(W0,lxd)


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
lxd = float64(lxd)
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


lx = float64(lxd/W0) # non-dimensionalize
lz = float64(aratio*lx)
nz = int32(aratio*nx+1)
print(aratio*nx)
nv= nz*nx #number of variables
dx = float64( lx/nx )
dz = float64( lz/(nz-1) ) 
x = (np.linspace(0,lx-dx,nx)).astype(np.float64)
z = (np.linspace(0,lz,nz)).astype(np.float64)
zz,xx = np.meshgrid(z, x)


dxdz_in = float64( 1./(dx*dz) ) 

hi= float64( 1./dx )

dt_sqrt =  float64( np.sqrt(dt) )

# Tishot = np.zeros((2*nv,nts+1))
order_param = np.zeros((nv,nts+1), dtype=np.float64)
conc = np.zeros((nv,nts+1), dtype=np.float64)

z_tip = 1 	# global variable that stores the tip z-index
tip_thres = np.int32(math.ceil(0.8*nz)) # threshold for moving the frame
print('threshold = ', tip_thres)


def update_tip(z_tip):

    checktip = cp.mean(cp.take(phi_cp, z_tip, 1))

    while checktip > -0.99:

        z_tip += 1
        checktip = cp.mean(cp.take(phi_cp, z_tip, 1))
    
    return z_tip

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
#@cuda.jit(device=True)
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
        return a_s
    
    
@cuda.jit('float64(float64, float64)',device=True)
#@cuda.jit(device=True)
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
def rhs_psi(ps, ph, U, ps_new, ph_new, U_new, zz, dpsi, nt, rng_states):
    # ps = psi, ph = phi

    i,j = cuda.grid(2)
    m,n = ps.shape
    
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

        
        Up = (zz[i,j] - R_tilde * (nt*dt) )/lT_tilde

        rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
                   sqrt2*ph[i,j] - lamd*(1-ph[i,j]**2)*sqrt2*(U[i,j] + Up)


        # =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================
        tp = (1-(1-k)*Up)
        tau_psi = tp*A2 if tp >= k else k*A2
        
        dpsi[i,j] = rhs_psi / tau_psi # + eta*(random()-0.5)/dt_sr
        
        #x = xoroshiro128p_uniform_float64(rng_states, thread_id)
        threadID = j*m + i
        beta_ij = xoroshiro128p_uniform_float64(rng_states, threadID) - 0.5 # rand from  [-0.5, 0.5]
        
        # update psi and phi
        ps_new[i,j] = ps[i,j] + ( dt * dpsi[i,j] + 0*dt_sqrt* eta * beta_ij ) 
        ph_new[i,j] = math.tanh(ps_new[i,j]/sqrt2)



@cuda.jit
def moveframe(ps, ph, U, zz, ps_buff, ph_buff, U_buff, zz_buff):

    i,j = cuda.grid(2)
    m,n = ps.shape

    if 0 < i < m-1 and 0 < j < n-2 :
        
        ps_buff[i,j] = ps[i,j+1]
        ph_buff[i,j] = ph[i,j+1]
        U_buff[i,j]  = U[i,j+1]
        zz_buff[i,j] = zz[i,j+1]

    if 0 < i < m-1 and j == n-2 :
        ps_buff[i,j] = 2*ps[i,n-2] - ps[i,n-3] # extrapalation
        ph_buff[i,j] = 2*ph[i,n-2] - ph[i,n-3]
        U_buff[i,j]  = U_0 
        zz_buff[i,j] = 2*zz[i,n-2] - zz[i,n-3]

@cuda.jit
def copyframe(ps_buff, ph_buff, U_buff, zz_buff, ps, ph, U, zz):

    i,j = cuda.grid(2)
    m,n = ps.shape

    if 0 < i < m-1 and 0 < j < n-1 :
        ps[i,j] = ps_buff[i,j]
        ph[i,j] = ph_buff[i,j]
        U[i,j]  = U_buff[i,j]
        zz[i,j] = zz_buff[i,j]


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


 
        
'''
@cuda.jit
def setBC_gpu(u):
    i,j = cuda.grid(2)
    m,n = u.shape
    
    
    if i == 0 and j < n :
        u[i,j] = u[m-2,j] # periodic
    
    if i == m-1 and j < n :
        u[i,j] = u[1,j] # periodic
     
    if j == 0 and i < m:
        u[i,j] = u[i,2] # no-flux
    
    if j == n-1 and i < m:
        u[i,j] = u[i, n-3] # no-flux 
'''

@cuda.jit
def setBC_all(ps,ph,U):
    i,j = cuda.grid(2)
    m,n = ps.shape
    
    
    if i == 0 and j < n :
        ps[i,j] = ps[m-2,j] # periodic
        ph[i,j] = ph[m-2,j] # periodic
        U[i,j]  = U[m-2,j] # periodic

    if i == m-1 and j < n :
        ps[i,j] = ps[1,j] # periodic
        ph[i,j] = ph[1,j] # periodic
        U[i,j]  = U[1,j] # periodic
     
    if j == 0 and i < m:
        ps[i,j] = ps[i,2] 
        ph[i,j] = ph[i,2]
        U[i,j]  = U[i,2] 

    if j == n-1 and i < m:
        ps[i,j] = ps[i,n-3] 
        ph[i,j] = ph[i,n-3]
        U[i,j]  = U[i,n-3] 


@cuda.jit
def rhs_U(U, U_new, ph, dpsi):


    i,j = cuda.grid(2)
    m,n = U.shape

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
       
        # dU[i,j] = rhs_U / tau_U

        U_new[i,j] = U[i,j] + dt * ( rhs_U / tau_U )


   

@cuda.jit
def update_phi_U(ps,ph,U,dU):

    i,j = cuda.grid(2)
    m,n = U.shape

    if  i < m and j < n :
        
        ph[i,j] = math.tanh( ps[i,j] / sqrt2)
        U[i,j] += dt * dU[i,j]



def save_data(phi,U):
    
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )
    
    return np.reshape(phi[1:-1,1:-1],     (nv,1), order='F') , \
           np.reshape(c_tilde[1:-1,1:-1], (nv,1), order='F') 

##############################################################################
#psi0 = PARA.sins_initial(lx,nx,xx,zz)
psi0 = PARA.seed_initial(xx,lx,zz)

#psi0 = PARA.planar_initial(lx,zz)
U0 = 0*psi0 + U_0
phi0 = np.tanh(psi0/sqrt2)

# append halo around data=
psi = set_halo(psi0)
phi = set_halo(phi0)
U = set_halo(U0)
zz = set_halo(zz)

# set BCs
set_BC(psi, 0, 1)
set_BC(U, 0, 1)
set_BC(phi,0,1)


phi_cpu = phi.astype(np.float64)
U_cpu = U.astype(np.float64)
psi_cpu = psi.astype(np.float64)


# save initial data
order_param[:,[0]], conc[:,[0]] = save_data(phi_cpu, U_cpu)


# arrays on device
psi_old = cuda.to_device(psi_cpu)
phi_old = cuda.to_device(phi_cpu)
U_old   = cuda.to_device(U_cpu)

psi_new = cuda.device_array_like(psi_old)
phi_new = cuda.device_array_like(phi_old)
U_new = cuda.device_array_like(U_old)

zz_gpu = cuda.to_device(zz)

dPSI = cuda.device_array(psi_cpu.shape, dtype=np.float64)
dU = cuda.device_array(U_cpu.shape, dtype=np.float64)

# buffer on GPU to move frame
psi_tmp = cuda.device_array(psi_cpu.shape, dtype=np.float64)
phi_tmp = cuda.device_array(phi_cpu.shape, dtype=np.float64)
U_tmp = cuda.device_array(U_cpu.shape, dtype=np.float64)
zz_tmp = cuda.device_array(zz.shape, dtype=np.float64)


# CUDA kernel invocation parameters
tpb2d = (16,16)
bpg_x = math.ceil(phi.shape[0] / tpb2d[0])
bpg_y = math.ceil(phi.shape[1] / tpb2d[1])
bpg2d = (bpg_x, bpg_y)

# CUDA random number states init
# seed_val = np.uint64(np.random.randint(1))
rng_states = create_xoroshiro128p_states( tpb2d[0]*tpb2d[1]*bpg_x*bpg_y, seed=seed_val)


tpb = 16 * 1
bpg = math.ceil( np.max([nx,nz]) / tpb )

print('(tpb,bpg) = ({0:2d},{1:2d})'.format(tpb, bpg))

kts = int(Mt/nts)
start2 = time.time()
# two steps per loop
for nt in range(int(Mt/2)):
   
    # =================================================================
    # 1. rhs of psi
    # =================================================================
    rhs_psi[bpg2d, tpb2d](psi_old, phi_old, U_old, psi_new, phi_new, U_new, zz_gpu, dPSI, 2*nt, rng_states)
    
 
    # =================================================================
    # 2.set BC of dPSI, U_old, phi_new, psi_new
    # ================================================================= 
#    setBC_gpu[ blockspergrid, threadsperblock ](dPSI_cp)
#    cuda.synchronize()
    setBC_gpu[bpg,tpb](psi_new, phi_new, U_old, dPSI)
    

    # =================================================================
    # 3.rhs of U, also update psi
    # ================================================================= 
    rhs_U[bpg2d, tpb2d](U_old, U_new, phi_old, dPSI)


    # 
    rhs_psi[bpg2d, tpb2d](psi_new, phi_new, U_new, psi_old, phi_old, U_old, zz_gpu, dPSI, 2*nt+1, rng_states)
    setBC_gpu[bpg,tpb](psi_old, phi_old, U_new, dPSI)
    rhs_U[bpg2d, tpb2d](U_new, U_old, phi_new, dPSI)
    
    if (2*nt+2)%kts==0:  # data saving
       kkk = int(np.floor((2*nt+2)/kts))
       phi = phi_old.copy_to_host()
       U  = U_old.copy_to_host()
       order_param[:,[kkk]], conc[:,[kkk]] = save_data(phi,U)
       


#    cuda.synchronize()

#    update_psi[ blockspergrid, threadsperblock ](psi_cp,dPSI_cp)
#    cuda.synchronize()

#    setBC_gpu[ blockspergrid, threadsperblock ](psi_cp)
#    cuda.synchronize()
    
    # =================================================================
    # 4. update U and phi
    # ================================================================= 
#    update_phi_U[ blockspergrid, threadsperblock ](psi_cp, phi_cp, U_cp, dU_cp)
#    cuda.synchronize()
  
    
    # =================================================================
    # 5.set BC of psi, phi, U
    # ================================================================= 
#     setBC_all[ blockspergrid, threadsperblock ](psi_cp, phi_cp, U_cp)
#    cuda.synchronize()

    # find tip position    
#    z_tip=update_tip(z_tip)
#    cuda.synchronize()    
 
    # when tip hit tip_thres, shift down by 1
#    if z_tip == tip_thres:

#        moveframe[ blockspergrid, threadsperblock ](psi_cp, phi_cp, U_cp, zz_cp, psi_buff, phi_buff, U_buff, zz_buff)
#        cuda.synchronize()

#        copyframe[ blockspergrid, threadsperblock ](psi_buff, phi_buff, U_buff, zz_buff, psi_cp, phi_cp, U_cp, zz_cp)
#        cuda.synchronize()

#        z_tip = z_tip-1


    # =================================================================
    # 5.set BC of psi, phi, U
    # =================================================================
#    setBC_all[ blockspergrid, threadsperblock ](psi_cp, phi_cp, U_cp)
#    cuda.synchronize()


    #setBC_gpu[ blockspergrid, threadsperblock ](phi_cp)
    #cuda.synchronize()

    #setBC_gpu[ blockspergrid, threadsperblock ](U_cp)
    #cuda.synchronize() 

#    cuda.synchronize()
#    if math.remainder(nt+1 , Mt/nts) == 0 :
#        print('time step = ', nt+1 )
#        print('tip position nz = ', z_tip)

#        phi = cp.asnumpy(phi_cp)
#        U = cp.asnumpy(U_cp)

#        print(phi.dtype)    

#        tk = np.int32((nt+1)/(Mt/nts))
#        order_param[:,[tk]], conc[:,[tk]] = save_data(phi,U) 

        # save data

end2=time.time()

#phi = phi_old.copy_to_host()
#U   = U_old.copy_to_host()
#order_param[:,[1]], conc[:,[1]] = save_data(phi,U)


print('elapse2: ', (end2-start2))

save(os.path.join(direc,filename),{'order_param':order_param, 'conc':conc, 'xx':xx*W0, 'zz':zz[1:-1,1:-1]*W0,'dt':dt*tau0, \
     'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end2-start2,'dPSI':dPSI.copy_to_host() } )
# save(os.path.join(direc,filename),{'xx':xx*W0,'zz':zz[1:-1,1:-1].T*W0,'y':Tishot,'dt':dt*tau0,'nx':nx,'nz':nz,'t':t*tau0,'mach_time':end-start})
