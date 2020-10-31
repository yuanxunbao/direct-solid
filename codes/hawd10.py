#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:10:13 2020

@author: yigong qin, yuanxun bao
"""
from mpi4py import MPI
import importlib
import sys
import os
import scipy.io as sio
from scipy.io import savemat as save
from numba import njit, cuda, vectorize, float64, float64, int32
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
delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m_slope, G, R, Ti, U_0, cl_0 = PARA.phys_para()
eps, alpha0, lx, aratio, nx, dt, Mt, eta, \
seed_val, nts,direc, mvf, tip_thres, ictype, qts,qoi_winds = PARA.simu_para(W0,Dl_tilde,tau0)

# dimensionalize
lxd = lx * W0

mph = 'cell' if eta ==0.0 else 'dendrite'

filename = 'DNS'+ '_G'+str('%4.2F'%G) + '_R'+str('%4.2F'%R) + '_noise'+ \
str('%4.2F'%eta)+'_misori'+str(alpha0)+'_lx'+ str('%4.2F'%lxd)+'_nx'+str('%d'%nx)+'_asp'+str(aratio)+ \
'_ictype'+ str('%d'%ictype) + '_U0'+str('%4.2F'%U_0)+'ha_wd10' 

# calculate snapshot / qoi to save
kts = int( 2*np.floor((Mt/nts)/2) ); # print(kts)
nts = int(Mt/kts); print(nts)
interq = int( 2*np.floor(Mt/qts/2) ); # print(interq)
qts = int(Mt/interq); print(qts)

'''
-------------------------------------------------------------------------------------------------
MPI ENVIRONMENT SETTING
-------------------------------------------------------------------------------------------------
'''

@cuda.jit
def setNBC_gpu(ps,ph,U,dpsi, px, py, nprocx, nprocy, ha_wd):

    m,n = ph.shape
    i = cuda.grid(1)
    # no flux  at x = 0, lx
    if ( px==nprocx-1 and i<n ):
        ps[-ha_wd,i]    = ps[-ha_wd-2,i]
        ph[-ha_wd,i]    = ph[-ha_wd-2,i]
        U[-ha_wd,i]     = U[-ha_wd-2,i]
        dpsi[-ha_wd,i]  = dpsi[-ha_wd-2,i]
    if ( px==0 and i<n):
        ps[ha_wd-1,i]   = ps[ha_wd+1,i]
        ph[ha_wd-1,i]   = ph[ha_wd+1,i]
        U[ha_wd-1,i]    = U[ha_wd+1,i]
        dpsi[ha_wd-1,i] = dpsi[ha_wd+1,i]
    # no flux  at z = 0, lz
    if ( py==0 and i<m):
        ps[i,ha_wd-1]   = ps[i,ha_wd+1]
        ph[i,ha_wd-1]   = ph[i,ha_wd+1]
        U[i,ha_wd-1]    = U[i,ha_wd+1]
        dpsi[i,ha_wd-1] = dpsi[i,ha_wd+1]
    if ( py==nprocy-1 and i<m):
        ps[i,-ha_wd]   = ps[i,-ha_wd-2]
        ph[i,-ha_wd]   = ph[i,-ha_wd-2]
        U[i,-ha_wd]    = U[i,-ha_wd-2]
        dpsi[i,-ha_wd] = dpsi[i,-ha_wd-2]

@cuda.jit
def BC_421(ps, ph, U, dpsi, BC): # size of ph is nx+2*ny+2, ph_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m- 2*ha_wd ;ny = n- 2*ha_wd
    i,j = cuda.grid(2)
    # the order: [0,:],[-1,:],[:,0],[:,-1]

    if (  i< ny and j < ha_wd ):
        BC[i,j+0*ha_wd] = ps[j+ha_wd,i+ha_wd]
        BC[i,j+1*ha_wd] = ph[j+ha_wd,i+ha_wd]
        BC[i,j+2*ha_wd] = U[j+ha_wd,i+ha_wd]
        BC[i,j+3*ha_wd] = dpsi[j+ha_wd,i+ha_wd]
        
        BC[i+ny,j+0*ha_wd] = ps[j+nx,i+ha_wd]
        BC[i+ny,j+1*ha_wd] = ph[j+nx,i+ha_wd]
        BC[i+ny,j+2*ha_wd] = U[j+nx,i+ha_wd]
        BC[i+ny,j+3*ha_wd] = dpsi[j+nx,i+ha_wd]

    if (  i < nx and j < ha_wd ):
        
        BC[i+2*ny,j+0*ha_wd] = ps[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+1*ha_wd] = ph[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+2*ha_wd] = U[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+3*ha_wd] = dpsi[i+ha_wd,j+ha_wd]

        BC[i+2*ny+nx,j+0*ha_wd] = ps[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+1*ha_wd] = ph[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+2*ha_wd] = U[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+3*ha_wd] = dpsi[i+ha_wd,j+ny]



@cuda.jit
def BC_124(ps, ph, U, dpsi, BC): # size of ph is nx*ny, nbh_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m- 2*ha_wd ;ny = n- 2*ha_wd
    i,j = cuda.grid(2)
    # the order: [0,:],[-1,:],[:,0],[:,-1]

    if (  i< ny and j < ha_wd ):
        ps[j,i+ha_wd]     = BC[i,j+0*ha_wd]
        ph[j,i+ha_wd]     = BC[i,j+1*ha_wd]
        U[j,i+ha_wd]      = BC[i,j+2*ha_wd]
        dpsi[j,i+ha_wd]   = BC[i,j+3*ha_wd]

        ps[j+nx+ha_wd,i+ha_wd]   = BC[i+ny,j+0*ha_wd]
        ph[j+nx+ha_wd,i+ha_wd]   = BC[i+ny,j+1*ha_wd]
        U[j+nx+ha_wd,i+ha_wd]    = BC[i+ny,j+2*ha_wd]
        dpsi[j+nx+ha_wd,i+ha_wd] = BC[i+ny,j+3*ha_wd]


    if (  i < nx and j < ha_wd ):

        ps[i+ha_wd,j]   = BC[i+2*ny,j+0*ha_wd]
        ph[i+ha_wd,j]   = BC[i+2*ny,j+1*ha_wd]
        U[i+ha_wd,j]    = BC[i+2*ny,j+2*ha_wd]
        dpsi[i+ha_wd,j] = BC[i+2*ny,j+3*ha_wd]

        ps[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+0*ha_wd]
        ph[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+1*ha_wd]
        U[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+2*ha_wd]
        dpsi[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+3*ha_wd]

   # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1]
    if i < ha_wd and j < ha_wd:
       ps[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+0*ha_wd];
       ph[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+1*ha_wd];
       U[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+2*ha_wd];
       dpsi[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+3*ha_wd];
       ps[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+0*ha_wd];
       ph[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+1*ha_wd];
       U[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+2*ha_wd];
       dpsi[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+3*ha_wd];

       ps[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+0*ha_wd];
       ph[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+1*ha_wd];
       U[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+2*ha_wd];
       dpsi[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+3*ha_wd];
       ps[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+0*ha_wd];
       ph[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+1*ha_wd];
       U[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+2*ha_wd];
       dpsi[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+3*ha_wd];


def BC_comm(sendbuf, recvbuf, nx ,ny):

    if ( px < nprocx-1 ):
           # right_send = phi_gpu[-2*ha_wd:-ha_wd,:].reshape((ha_wd,ny+2*ha_wd))
            comm.Send([sendbuf[ny:2*ny,:],MPI.DOUBLE], dest = rank+1, tag=1)

    if ( px > 0 ):
            comm.Recv([recvbuf[:ny,:],MPI.DOUBLE],source = rank-1, tag=1)
           # phi_gpu[:ha_wd,:] = left_recv
    # sending direction:  X decreases
    if ( px > 0 ):
            comm.Send([sendbuf[:ny,:],MPI.DOUBLE], dest = rank-1, tag=2)
    if ( px < nprocx-1 ):
            comm.Recv([recvbuf[ny:2*ny,:],MPI.DOUBLE],source = rank+1, tag=2)

    # sending direction:  Y(Z) increases
    if ( py < nprocy-1 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+nprocx, tag=3)
    if ( py>0 ):
            comm.Recv([recvbuf[2*ny:2*ny+nx,:],MPI.DOUBLE],source = rank-nprocx, tag=3)
    # sending direction:  Y(Z) decreases
    if ( py >0 ):
            comm.Send([sendbuf[2*ny:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx, tag=4)
    if ( py < nprocy -1 ):
            comm.Recv([recvbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE],source = rank +nprocx, tag=4)


        # send and recv corner data, the order of send is [-1,-1],[0,-1],[-1,0],[0,0]
        # the order: [0,:],[-1,:],[:,0],[:,-1]
        # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1] send the corner data from [:,0] [:,-1]
    if ( px < nprocx-1 and py < nprocy-1):
            comm.Send([sendbuf[2*ny+2*nx-ha_wd:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+1+nprocx, tag=5)

    if ( px > 0 and py > 0 ):
            comm.Recv([recvbuf[-4*ha_wd:-3*ha_wd,:],MPI.DOUBLE],source = rank-1-nprocx, tag=5)
    # sending direction:  X decreases    send [0,0] recv [-1,-1]
    if ( px > 0 and py > 0):
            comm.Send([sendbuf[2*ny:2*ny+ha_wd,:],MPI.DOUBLE], dest = rank-1-nprocx, tag=6)
    if ( px < nprocx-1 and py < nprocy-1 ):
            comm.Recv([recvbuf[-ha_wd:,:],MPI.DOUBLE],source = rank+1+nprocx, tag=6)

    # sending direction:  Y(Z) increases  send [0,-1] recv [-1,0]
    if ( py < nprocy-1 and px > 0 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+nx+ha_wd,:],MPI.DOUBLE], dest = rank+nprocx-1, tag=7)
    if ( py>0 and px < nprocx-1 ):
            comm.Recv([recvbuf[-3*ha_wd:-2*ha_wd,:],MPI.DOUBLE],source = rank-nprocx+1, tag=7)
    # sending direction:  Y(Z) decreases  send [-1,0] recv [0,-1]
    if ( py>0 and px < nprocx-1):
            comm.Send([sendbuf[2*ny+nx-ha_wd:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx+1, tag=8)
    if ( py < nprocy -1 and px > 0):
            comm.Recv([recvbuf[-2*ha_wd:-ha_wd,:],MPI.DOUBLE],source = rank +nprocx-1, tag=8)

    return


comm = MPI.COMM_WORLD           # initialize MPI
rank = comm.Get_rank()          # id of the current processor [0:nproc]
nproc = comm.Get_size()         # number of processors

num_gpus_per_node = 4
gpu_name = cuda.select_device( rank % num_gpus_per_node)

if rank == 0: print('GPUs on this node', cuda.gpus)
print('device id',gpu_name,'host id',rank )

nprocx = int(np.ceil(np.sqrt(nproc)))
nprocy = int(np.ceil(nproc/nprocx))


px = rank%nprocx           # x id of current processor   [0:nprocx]
py = int(np.floor(rank/nprocx)) # y id of current processor  [0:nprocy]
print('px ',px,'py ',py,' for rank ',rank)

if rank ==0: print('total/x/y processors', nproc, nprocx, nprocy)







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
#alpha0_rad = float64(alpha0*math.pi/180)
lx = float64(lx)
nx = int32(nx)
dt = float64(dt)
Mt = int32(Mt)
eta = float64(eta)
U_0 = float64(U_0)

#cosa = float64(np.cos(alpha0_rad))
#sina = float64(np.sin(alpha0_rad))        #comment out the universal misorientation angle  !!!!!!!!!!!!!!!!!!!!!!!!

a_s = float64(1 - 3*delta)
epsilon = float64(4.0*delta/a_s)
a_12 = float64(4.0*a_s*epsilon)


sqrt2 = float64(np.sqrt(2.0))

#print('change all boundary conditions to Neumann boundary condtion ============================')
filename = filename + 'DNS' +'rank'+str(rank)
lz = float64(aratio*lx)
len_blockx = lx/nprocx
len_blockz = lz/nprocy

new_ratio = int32(len_blockz/len_blockx)
print('aspect ratio',new_ratio, 'for rank',rank)
lminx = px*len_blockx
lminz = py*len_blockz

dx = 1.2
nx = int32(len_blockx/dx)
nz = int32(new_ratio*nx)

dx = float64( len_blockx/nx )
dz = float64( len_blockz/nz )

# nx and nz should be the smaller ones
if px == 0:
    if py == 0:
        nx = nx+1; nz=nz+1;
        x_1d = np.linspace(lminx,lminx+len_blockx,num=nx)
        z_1d = np.linspace(lminz,lminz+len_blockz,num=nz)
    else:
        nx = nx+1; nz=nz;
        x_1d = np.linspace(lminx,lminx+len_blockx,num=nx)
        z_1d = np.linspace(lminz+dz,lminz+len_blockz,num=nz)

elif px > 0 and py == 0:
    nx = nx; nz=nz+1;
    x_1d = np.linspace(lminx+dx,lminx+len_blockx,num=nx)
    z_1d = np.linspace(lminz,lminz+len_blockz,num=nz)

elif px > 0 and py > 0:
    nx = nx; nz=nz;
    x_1d = np.linspace(lminx+dx,lminx+len_blockx,num=nx)
    z_1d = np.linspace(lminz+dz,lminz+len_blockz,num=nz)

else: print('invalid proceesor ID occurs!')

print('rank ',rank, ' x range ', x_1d[0], x_1d[-1], 'nx', nx ,'dx', dx)
print('rank ',rank, ' z range ', z_1d[0], z_1d[-1], 'nz', nz ,'dz', dz)
ha_wd = int(sys.argv[3])
nv= (nz)*(nx) #number of variables

#print('change all boundary conditions to Neumann boundary condtion ============================')

#lz = float64(aratio*lx)
#nz = int32(aratio*(nx-1)+1)               # make change about BC here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#nv= nz*nx #number of variables
#dx = float64( lx/(nx-1) )                # make change about BC here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#dz = float64( lz/(nz-1) ) 
#
#x_1d = (np.linspace(0,lx,nx)).astype(np.float64)     # make change about BC here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#z_1d = (np.linspace(0,lz,nz)).astype(np.float64)
zz,xx = np.meshgrid(z_1d, x_1d)

dxdz_in = float64( 1./(dx*dz) ) 
dxdz_in_sqrt = float64( np.sqrt(dxdz_in) )

hi= float64( 1./dx )

dt_sqrt =  float64( np.sqrt(dt) )

dxd = dx*W0

if rank == 0:
  print('==========================================\n')
  print('W0 = %4.2E um'% W0)
  print('tau0 = %4.12E s'%tau0)
  #print('dx = %4.2E um'%(dx*W0))
  print('dt = %4.12E s'%(dt*tau0))
  print('lambda = ', lamd)
  print('Mt = ', Mt)
  print('U0 = ', U_0)
  #print('grid = %d x %d'%(nx,nz))
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
    ub = np.zeros((m+ 2*ha_wd,n+ 2*ha_wd))

    ub[ha_wd:-ha_wd,ha_wd:-ha_wd] = u

    return ub


'''
Device function
'''
@cuda.jit('float64(float64, float64, float64, float64)',device=True)
def atheta(ux, uz, cosa, sina):

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
    
    
@cuda.jit('float64(float64, float64, float64, float64)',device=True)
def aptheta(ux, uz, cosa, sina):
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
def rhs_psi(ps, ph, U, ps_new, ph_new, U_new, zz, dpsi, nt, rng_states, T_m, alpha_m):
    # ps = psi, ph = phi

    i,j = cuda.grid(2)
    m,n = ps.shape
    
    # thread on interior points
    if 0 < i < m-1 and 0 < j < n-1:

        alpha = alpha_m[i,j]
        cosa = math.cos(alpha) 
        sina = math.sin(alpha)

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

        A  = atheta( phx,phz, cosa, sina)
        Ap = aptheta(phx,phz, cosa, sina)
        JR = A * ( A*psx - Ap*psz )
        
        # ============================
        # left edge flux
        # ============================
        psx = ps[i+0,j+0]-ps[i-1,j+0]
        psz = psimjp - psimjm
        phx = ph[i+0,j+0]-ph[i-1,j+0]
        phz = phimjp - phimjm

        A  = atheta( phx,phz, cosa, sina)
        Ap = aptheta(phx,phz, cosa, sina)
        JL = A * ( A*psx - Ap*psz )
        
        # ============================
        # top edge flux
        # ============================
        psx = psipjp - psimjp
        psz = ps[i+0,j+1]-ps[i+0,j+0]
        phx = phipjp - phimjp
        phz = ph[i+0,j+1]-ph[i+0,j+0]

        A  = atheta( phx,phz, cosa, sina)
        Ap = aptheta(phx,phz, cosa, sina)
        JT = A * ( A*psz + Ap*psx )

        # ============================
        # bottom edge flux
        # ============================
        psx = psipjm - psimjm
        psz = ps[i+0,j+0]-ps[i+0,j-1]
        phx = phipjm - phimjm
        phz = ph[i+0,j+0]-ph[i+0,j-1]

        A  = atheta( phx,phz, cosa, sina)
        Ap = aptheta(phx,phz, cosa, sina)
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

        A2 = atheta(phxn,phzn, cosa, sina)**2
        gradps2 = (psxn)**2 + (pszn)**2
        extra =  -sqrt2 * A2 * ph[i,j] * gradps2


        # =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================

        # print(lT_tilde)         
        #Up = (zz[j] - R_tilde * (nt*dt) )/lT_tilde
        Up = ( T_m[i,j] - Ti )/( m_slope*c_inf/k )/(1-k)
        
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
#    if ( i < n ):
#        ps[0,i]   = ps[m-2,i]
#        ph[0,i]   = ph[m-2,i]
#        U[0,i]    = U[m-2,i]
#        dpsi[0,i] = dpsi[m-2,i]
#
#        ps[m-1,i]   = ps[1,i]
#        ph[m-1,i]   = ph[1,i]
#        U[m-1,i]    = U[1,i]
#        dpsi[m-1,i] = dpsi[1,i]
    if ( i < n ):
        ps[0,i]   = ps[2,i]
        ph[0,i]   = ph[2,i]
        U[0,i]    = U[2,i]
        dpsi[0,i] = dpsi[2,i]

        ps[m-1,i]   = ps[m-3,i]
        ph[m-1,i]   = ph[m-3,i]
        U[m-1,i]    = U[m-3,i]
        dpsi[m-1,i] = dpsi[m-3,i]

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


@cuda.jit
def XYT_lin_interp(x, y, t, X, Y, T,  u_3d,u_m,v_3d,v_m ):
    # array description: x (nx,); y (ny,); t (scalar); X (Nx,); Y (Ny,); T (Nt,); u_3d (Nx, Ny, Nt); u_2d (Nx,Ny)
    # note that u_m, v_m here have halo, so dimension is (nx+2, ny+2)
    i,j = cuda.grid(2)
    Nx,Ny,Nt = u_3d.shape
    m, n = u_m.shape  #(m = nx+2, n = ny +2)
    nx = m -2*ha_wd; ny = n -2*ha_wd;
    # first step is get the id of the time kt and kt+1, do a interpation between kt layer and kt+1 layer first
    Dt = T[1] - T[0]
    kt = int( ( t - T[0] )/Dt )
    delta_t = ( t - T[0] )/Dt - kt

    # then preform a standard 2d interplation from Nx, Ny to nx+2, ny+2
    Dx = X[1] - X[0]; Dy = Y[1] - Y[0]  # for now assume uniform mesh

    if  i < m and j < n :

        kx = int( ( x[i] - X[0] )/Dx )
        delta_x = ( x[i] - X[0] )/Dx - kx

        ky = int( ( y[j] - Y[0] )/Dy )
        delta_y = ( y[j] - Y[0] )/Dy - ky

        u_m[i,j] = ( (1-delta_x)*(1-delta_y)*u_3d[kx,ky,kt] + (1-delta_x)*delta_y*u_3d[kx,ky+1,kt] \
                     +delta_x*(1-delta_y)*u_3d[kx+1,ky,kt] +   delta_x*delta_y*u_3d[kx+1,ky+1,kt] )*(1-delta_t) + \
                   ( (1-delta_x)*(1-delta_y)*u_3d[kx,ky,kt+1] + (1-delta_x)*delta_y*u_3d[kx,ky+1,kt+1] \
                     +delta_x*(1-delta_y)*u_3d[kx+1,ky,kt+1] +   delta_x*delta_y*u_3d[kx+1,ky+1,kt+1] )*delta_t

        v_m[i,j] = ( (1-delta_x)*(1-delta_y)*v_3d[kx,ky,kt] + (1-delta_x)*delta_y*v_3d[kx,ky+1,kt] \
                     +delta_x*(1-delta_y)*v_3d[kx+1,ky,kt] +   delta_x*delta_y*v_3d[kx+1,ky+1,kt] )*(1-delta_t) + \
                   ( (1-delta_x)*(1-delta_y)*v_3d[kx,ky,kt+1] + (1-delta_x)*delta_y*v_3d[kx,ky+1,kt+1] \
                     +delta_x*(1-delta_y)*v_3d[kx+1,ky,kt+1] +   delta_x*delta_y*v_3d[kx+1,ky+1,kt+1] )*delta_t

    return

def save_data(phi,U,z):
    
    cinf_cl0 =  1+ (1-k)*U_0
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 ) / cinf_cl0
    
#    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )
   
    return np.reshape(phi[ha_wd:-ha_wd,ha_wd:-ha_wd],     (nv,1), order='F') , \
           np.reshape(c_tilde[ha_wd:-ha_wd,ha_wd:-ha_wd], (nv,1), order='F') , \
           z[ha_wd:-ha_wd,].T

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
    
    # construct a radial distribution of U and psi
  if len(sys.argv)==4:
    dd = sio.loadmat(sys.argv[3],squeeze_me = True)

    #tr_tip = dd['trans_tip']
    if rank == 0:
      tr_tip = 0
      psi0 = np.repeat( (dd['op_psi_1d'])[tr_tip:nz+tr_tip,-1].reshape((1,nz)), nx, axis=0)
      U0 = np.repeat( (dd['Uc_1d'])[tr_tip:nz+tr_tip,-1].reshape((1,nz)), nx, axis=0)
    else: 
      psi0 = -1*np.ones((nx,nz))
      U0 = -1*np.ones((nx,nz))
    phi0 = np.tanh(psi0/sqrt2)
    zz += dd['z_1d'][0]
   # Ti = dd['Ttip'][0]
  else: print('need macro data input!!!')
 
else: 
    print('ERROR: invalid type of initial condtion...' )
    sys.exit(1)


# append halo around data
psi = set_halo(psi0)
phi = set_halo(phi0)
U = set_halo(U0)
zz = set_halo(zz)
print(z_1d)
x_cpu = np.zeros(nx+2*ha_wd); x_cpu[ha_wd:-ha_wd] = x_1d
z_cpu = np.zeros(nz+2*ha_wd); z_cpu[ha_wd:-ha_wd] = z_1d
#z_cpu  = zz[ha_wd,:]
#x_cpu = set_halo(xx)[:,ha_wd]       # add x array here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# set BCs
setBC_cpu(psi, 1, 1)
setBC_cpu(U, 1, 1)
setBC_cpu(phi,1,1)


phi_cpu = phi.astype(np.float64)
U_cpu = U.astype(np.float64)
psi_cpu = psi.astype(np.float64)


# save initial data
op_phi[:,[0]], conc[:,[0]], zz_mv[:,0] = save_data(phi_cpu, U_cpu, z_cpu )




# allocate space on device
psi_old = cuda.to_device(psi_cpu)
phi_old = cuda.to_device(phi_cpu)
U_old   = cuda.to_device(U_cpu)

psi_new = cuda.device_array_like(psi_old)
phi_new = cuda.device_array_like(phi_old)
U_new = cuda.device_array_like(U_old)

T_m = cuda.device_array_like(U_old)
alpha_m = cuda.device_array_like(U_old)

dPSI = cuda.device_array(psi_cpu.shape, dtype=np.float64)

x_cpu[:ha_wd] = x_cpu[ha_wd:2*ha_wd]-ha_wd*dx; x_cpu[-ha_wd:] = x_cpu[-2*ha_wd:-ha_wd]+ha_wd*dx;
z_cpu[:ha_wd] = z_cpu[ha_wd:2*ha_wd]-ha_wd*dz; z_cpu[-ha_wd:] = z_cpu[-2*ha_wd:-ha_wd]+ha_wd*dz;
z_gpu  = cuda.to_device(z_cpu)
z_gpu_buff = cuda.to_device(z_cpu)
x_gpu  = cuda.to_device(x_cpu)      # add x array here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print('x',x_cpu)
#print('z',z_cpu)
#z_gpu_inte  = cuda.to_device(z_cpu[ha_wd:-ha_wd])

mac_data = sio.loadmat(sys.argv[2],squeeze_me = True)
X_gpu = cuda.to_device(mac_data['x_grid']/W0)
Z_gpu = cuda.to_device(mac_data['y_grid']/W0)
mac_t_gpu = cuda.to_device(mac_data['time_traj'])
T_3D_gpu = cuda.to_device(mac_data['T_arr'])

#if px ==0:coe_a = -2*0.1
#elif px ==1:coe_a = -0.1
#elif px ==2:coe_a = 0.1
#elif px ==3: coe_a = 2*0.1
#else: print('not enough processor in x direction')
alpha_3D_cpu = mac_data['alpha_arr']

alpha_3D_gpu = cuda.to_device(alpha_3D_cpu)


#print('x_1d',x_cpu)
#print('X_1d',mac_data['x_grid']/W0)
#print('z_1d',z_cpu[1:-1])
#print('Z_1d',mac_data['y_grid']/W0)
#print('time_1d',mac_data['time_traj'])
#print('T_2d',mac_data['T_arr'][-3:,-3:,0])
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

inter_len = np.zeros(qts); pri_spac = np.zeros(qts); sec_spac = np.zeros(qts);
fs_win = qoi_winds
fs_arr = np.zeros((fs_win,qts)); ztip_arr = np.zeros(qts);cqois = np.zeros((10,qts));
HCS = np.zeros(qts);Kc_ave = np.zeros(qts)
Ttip_arr = np.zeros(qts);
tip_uq = np.zeros(qts);
alpha_arr = np.zeros((nz,qts));

# array holds tip z-coord
ztip_qoi = np.zeros(qts)
time_qoi = np.zeros(qts)
tip_vel = np.zeros(qts)

# allocate boundary data
BCsend = cuda.device_array([2*nx+2*nz,4*ha_wd],dtype=np.float64);
BCrecv = cuda.device_array([2*nx+2*nz+4*ha_wd,4*ha_wd],dtype=np.float64)
#bpgBC = (2*bpg_x+2*bpg_y,1)
bpgBC = (2*(bpg_x+bpg_y),math.ceil(4*ha_wd/tpb) )
BC_421[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, BCsend)
#if rank == 0: print('sendbuf',BCsend.copy_to_host())
comm.Barrier()
BC_comm(BCsend, BCrecv, nx ,nz)
comm.Barrier()
BC_124[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, BCrecv)
BC_124[bpgBC,tpb2d](psi_new,phi_new,U_new, dPSI, BCrecv)
#if rank == 0: print('recvbuf',BCrecv.copy_to_host())
setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, px, py, nprocx, nprocy, ha_wd)
setNBC_gpu[bpg,tpb](psi_new,phi_new,U_old,dPSI, px, py, nprocx, nprocy, ha_wd)
#phi = U_old.copy_to_host()
#print('rank',rank,phi)
#if rank==0: print('rank',rank,phi)
# march two steps per loop
gpu_com=0
start = time.time()
for kt in range(100):#int(Mt/2)):
   
    # =================================================================
    # time step: t = (2*nt) * dt
    # =================================================================
    t_cur = (2*kt)*dt*tau0
    XYT_lin_interp[bpg2d, tpb2d](x_gpu, z_gpu, t_cur, X_gpu, Z_gpu, mac_t_gpu, T_3D_gpu, T_m, alpha_3D_gpu, alpha_m )
    rhs_psi[bpg2d, tpb2d](psi_old, phi_old, U_old, psi_new, phi_new, U_new, z_gpu, dPSI, 2*kt, rng_states, T_m, alpha_m)
  #  setBC_gpu[bpg,tpb](psi_new, phi_new, U_old, dPSI)
   # if rank==0:
   #   if ha_wd == 2: print(T_m.copy_to_host())
   #   else: print(T_m.copy_to_host())
    if ha_wd==1:
      BC_421[bpgBC,tpb2d](psi_new, phi_new, U_old, dPSI, BCsend)
      comm.Barrier()
      BC_comm(BCsend, BCrecv, nx ,nz)
      comm.Barrier()
      BC_124[bpgBC,tpb2d](psi_new, phi_new, U_old, dPSI, BCrecv)
#    if kt==1:
 #    if ha_wd == 2: print('rak',rank,U_old.copy_to_host())
  #   else: print(psi_new.copy_to_host())
    setNBC_gpu[bpg,tpb](psi_new,phi_new,U_old,dPSI, px, py, nprocx, nprocy, ha_wd)
    rhs_U[bpg2d, tpb2d](U_old, U_new, phi_old, dPSI)
  #  if rank==0:
   #     phi = T_m.copy_to_host()
    #    print('x',x_gpu.copy_to_host())
     #   print('z',z_gpu_inte.copy_to_host())
      #  print('rank',rank,phi[ha_wd:-ha_wd,ha_wd:-ha_wd])
  #  if kt ==0:
 #      T_cpu = T_m.copy_to_host()
#       print(T_cpu[-2,-2])
    # =================================================================
    # time step: t = (2*nt+1) * dt
    # =================================================================
    t_cur = (2*kt+1)*dt*tau0 
    XYT_lin_interp[bpg2d, tpb2d](x_gpu, z_gpu, t_cur, X_gpu, Z_gpu, mac_t_gpu, T_3D_gpu, T_m, alpha_3D_gpu, alpha_m )
    rhs_psi[bpg2d, tpb2d](psi_new, phi_new, U_new, psi_old, phi_old, U_old, z_gpu, dPSI, 2*kt+1, rng_states, T_m, alpha_m)
   # setBC_gpu[bpg,tpb](psi_old, phi_old, U_new, dPSI)  
#    setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, px, py, nprocx, nprocy, ha_wd)
 #   rhs_U[bpg2d, tpb2d](U_new, U_old, phi_new, dPSI)
 
    if (2*kt+2)%ha_wd==0: #ha_wd==1 or ha_wd==2:
      gpu_com +=1
      BC_421[bpgBC,tpb2d](psi_old,phi_old,U_new, dPSI, BCsend)
      comm.Barrier()
      BC_comm(BCsend, BCrecv, nx ,nz)
      comm.Barrier()
      BC_124[bpgBC,tpb2d](psi_old,phi_old,U_new, dPSI, BCrecv)
     # setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, px, py, nprocx, nprocy, ha_wd)
     # setNBC_gpu[bpg,tpb](psi_new,phi_new,U_old,dPSI, px, py, nprocx, nprocy, ha_wd)
   
      BC_421[bpgBC,tpb2d](psi_old,phi_new,U_new, dPSI, BCsend)
      comm.Barrier()
      BC_comm(BCsend, BCrecv, nx ,nz)
      comm.Barrier()
      BC_124[bpgBC,tpb2d](psi_old,phi_new,U_new, dPSI, BCrecv)
      setNBC_gpu[bpg,tpb](psi_old,phi_new,U_new,dPSI, px, py, nprocx, nprocy, ha_wd)
      print('GPU communication',gpu_com)
    setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, px, py, nprocx, nprocy, ha_wd)
    rhs_U[bpg2d, tpb2d](U_new, U_old, phi_new, dPSI) 
#    if (2*kt+2)%ha_wd==0: #ha_wd==1 or ha_wd==2:
 #     BC_421[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, BCsend)
  #    comm.Barrier()
   #   BC_comm(BCsend, BCrecv, nx ,nz)
    #  comm.Barrier()
     # BC_124[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, BCrecv)   
    if (2*kt+2)%kts==0:
       
       print('time step = ', 2*(kt+1) )
       if mvf == True: print('tip position nz = ', cur_tip)


       kk = int(np.floor((2*kt+2)/kts))
       phi = phi_old.copy_to_host()
       U  = U_old.copy_to_host()
       psi = psi_old.copy_to_host()
       z_cpu = z_gpu.copy_to_host()
       # Ttip_arr[kk] = Ti + G*( zz_cpu[3,cur_tip]*W0 - R*(2*nt+2)*dt*tau0 ) 
       op_phi[:,[kk]], conc[:,[kk]], zz_mv[:,kk] = save_data(phi,U,z_cpu)       
      # op_psi_1d[:,kk],op_phi_1d[:,kk], Uc_1d[:,kk],conc_1d[:,kk], z_1d[:,kk] = save_data_transient(psi, phi, U, z_cpu)

       t_snapshot[kk] = 2*(kt+1)*dt 
if rank==0 or rank==1:
  phi = phi_new.copy_to_host();  U = U_old.copy_to_host();
# if ha_wd ==1: print('rank',rank,phi)#[ha_wd:-ha_wd,ha_wd:-ha_wd])
  #if ha_wd ==1: print('rank',rank,U[1:-1,1:-1])
 # if ha_wd ==2: print('rank',rank,U[2:-2,2:-2])
  print('rank',rank,U[ha_wd:-ha_wd,ha_wd:-ha_wd])
end = time.time()
print('elapsed time: ', (end-start))

save(os.path.join(direc,filename+'.mat'),{'op_phi':op_phi, 'conc':conc, 'xx':xx*W0, 'zz_mv':zz_mv*W0,'dt':dt*tau0,\
 'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start, 't_snapshot':t_snapshot*tau0} )

'''
save(os.path.join(direc,filename+'_QoIs.mat'),{'time_qoi':time_qoi, 'ztip_qoi':ztip_qoi-ztip_qoi[0],\
'Ttip_arr':Ttip_arr,'tip_uq':tip_uq,'cqois':cqois,'pri_spac':pri_spac,'sec_spac':sec_spac,'interfl':inter_len,\
'fs_arr':fs_arr,'HCS':HCS,'Kc_ave':Kc_ave})

# save('initial.mat',{'phi_ic':phi, 'U_ic':U, 'psi_ic':psi, 'tip_x':cur_tip_x, 'tip_z':cur_tip, 'zz_mv':zz_mv})

save(os.path.join(direc,filename),{'order_param':order_param, 'conc':conc, 'xx':xx*W0, 'zz_mv':zz_mv*W0,'dt':dt*tau0,\
 'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start,'ztip':ztip_arr,'Tip':Ttip_arr,'inter_len':inter_len,'pri_spac':pri_spac,\
    'sec_spac':sec_spac,'alpha':alpha_arr,'fs':fs_arr } )
'''
