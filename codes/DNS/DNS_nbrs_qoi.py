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
from scipy.interpolate import griddata, interp1d, interp2d

PARA = importlib.import_module(sys.argv[1])
# import dsinput as PARA

'''
-------------------------------------------------------------------------------------------------
LOAD PARAMETERS
-------------------------------------------------------------------------------------------------
'''
delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m_slope, G, R, Ti, U_0, cl_0 = PARA.phys_para()
eps, alpha0, lx, aratio, nx, dt, Mt, eta, \
seed_val, nts,direc, mvf, tip_thres, ictype, qts,qoi_winds ,xmin_mic,zmin_mic,dx= PARA.simu_para(W0,Dl_tilde,tau0,sys.argv[2])

# dimensionalize
lxd = lx * W0

### liquid to solid transition ####
l2s = -0.995

mph = 'cell' if eta ==0.0 else 'dendrite'

filename = 'DNSalpha_comm_nbrs'+ (sys.argv[2])[:-4] + '_noise'+ \
str('%4.2F'%eta)+'_misori'+str(alpha0)+'_lx'+ str('%4.2F'%lxd)+'_nx'+str('%d'%nx)+'_asp'+str(aratio)+ \
'_ictype'+ str('%d'%ictype) + '_U0'+str('%4.2F'%U_0)+'seed'+str(seed_val) 

# calculate snapshot / qoi to save
kts = int( 2*np.floor((Mt/nts)/2) ); # print(kts)
nts = int(Mt/kts); #print(nts)
interq = int( 2*np.floor(Mt/qts/2) );  print('the interval of qoi',interq)
qts = int(Mt/interq); #print('the ',qts)

'''
-------------------------------------------------------------------------------------------------
MPI ENVIRONMENT SETTING
-------------------------------------------------------------------------------------------------
'''

@cuda.jit
def setNBC_gpu(ps,ph,U,dpsi,ph2, px, py, nprocx, nprocy, ha_wd):

    m,n = ph.shape
    i = cuda.grid(1)
    # no flux  at x = 0, lx
    if ( px==nprocx-1 and i<n ):
        ps[-ha_wd,i]    = ps[-ha_wd-2,i]
        ph[-ha_wd,i]    = ph[-ha_wd-2,i]
        U[-ha_wd,i]     = U[-ha_wd-2,i]
        dpsi[-ha_wd,i]  = dpsi[-ha_wd-2,i]
        ph2[-ha_wd,i]    = ph2[-ha_wd-2,i]
    if ( px==0 and i<n):
        ps[ha_wd-1,i]   = ps[ha_wd+1,i]
        ph[ha_wd-1,i]   = ph[ha_wd+1,i]
        U[ha_wd-1,i]    = U[ha_wd+1,i]
        dpsi[ha_wd-1,i] = dpsi[ha_wd+1,i]
        ph2[ha_wd-1,i]   = ph2[ha_wd+1,i]
    # no flux  at z = 0, lz
    if ( py==0 and i<m):
        ps[i,ha_wd-1]   = ps[i,ha_wd+1]
        ph[i,ha_wd-1]   = ph[i,ha_wd+1]
        U[i,ha_wd-1]    = U[i,ha_wd+1]
        dpsi[i,ha_wd-1] = dpsi[i,ha_wd+1]
        ph2[i,ha_wd-1]   = ph2[i,ha_wd+1]
    if ( py==nprocy-1 and i<m):
        ps[i,-ha_wd]   = ps[i,-ha_wd-2]
        ph[i,-ha_wd]   = ph[i,-ha_wd-2]
        U[i,-ha_wd]    = U[i,-ha_wd-2]
        dpsi[i,-ha_wd] = dpsi[i,-ha_wd-2]
        ph2[i,-ha_wd]   = ph2[i,-ha_wd-2]

@cuda.jit
def BC_421(ps, ph, U, dpsi, ph2, BC): # size of ph is nx+2*ny+2, ph_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m- 2*ha_wd ;ny = n- 2*ha_wd
    i,j = cuda.grid(2)
    # the order: [0,:],[-1,:],[:,0],[:,-1]

    if (  i< ny and j < ha_wd ):
        BC[i,j+0*ha_wd] = ps[j+ha_wd,i+ha_wd]
        BC[i,j+1*ha_wd] = ph[j+ha_wd,i+ha_wd]
        BC[i,j+2*ha_wd] = U[j+ha_wd,i+ha_wd]
        BC[i,j+3*ha_wd] = dpsi[j+ha_wd,i+ha_wd]
        BC[i,j+4*ha_wd] = ph2[j+ha_wd,i+ha_wd]
        
        BC[i+ny,j+0*ha_wd] = ps[j+nx,i+ha_wd]
        BC[i+ny,j+1*ha_wd] = ph[j+nx,i+ha_wd]
        BC[i+ny,j+2*ha_wd] = U[j+nx,i+ha_wd]
        BC[i+ny,j+3*ha_wd] = dpsi[j+nx,i+ha_wd]
        BC[i+ny,j+4*ha_wd] = ph2[j+nx,i+ha_wd]

    if (  i < nx and j < ha_wd ):
        
        BC[i+2*ny,j+0*ha_wd] = ps[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+1*ha_wd] = ph[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+2*ha_wd] = U[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+3*ha_wd] = dpsi[i+ha_wd,j+ha_wd]
        BC[i+2*ny,j+4*ha_wd] = ph2[i+ha_wd,j+ha_wd]

        BC[i+2*ny+nx,j+0*ha_wd] = ps[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+1*ha_wd] = ph[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+2*ha_wd] = U[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+3*ha_wd] = dpsi[i+ha_wd,j+ny]
        BC[i+2*ny+nx,j+4*ha_wd] = ph2[i+ha_wd,j+ny]


@cuda.jit
def BC_124(ps, ph, U, dpsi, ph2, BC): # size of ph is nx*ny, nbh_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m- 2*ha_wd ;ny = n- 2*ha_wd
    i,j = cuda.grid(2)
    # the order: [0,:],[-1,:],[:,0],[:,-1]

    if (  i< ny and j < ha_wd ):
        ps[j,i+ha_wd]     = BC[i,j+0*ha_wd]
        ph[j,i+ha_wd]     = BC[i,j+1*ha_wd]
        U[j,i+ha_wd]      = BC[i,j+2*ha_wd]
        dpsi[j,i+ha_wd]   = BC[i,j+3*ha_wd]
        ph2[j,i+ha_wd]     = BC[i,j+4*ha_wd]

        ps[j+nx+ha_wd,i+ha_wd]   = BC[i+ny,j+0*ha_wd]
        ph[j+nx+ha_wd,i+ha_wd]   = BC[i+ny,j+1*ha_wd]
        U[j+nx+ha_wd,i+ha_wd]    = BC[i+ny,j+2*ha_wd]
        dpsi[j+nx+ha_wd,i+ha_wd] = BC[i+ny,j+3*ha_wd]
        ph2[j+nx+ha_wd,i+ha_wd]   = BC[i+ny,j+4*ha_wd]

    if (  i < nx and j < ha_wd ):

        ps[i+ha_wd,j]   = BC[i+2*ny,j+0*ha_wd]
        ph[i+ha_wd,j]   = BC[i+2*ny,j+1*ha_wd]
        U[i+ha_wd,j]    = BC[i+2*ny,j+2*ha_wd]
        dpsi[i+ha_wd,j] = BC[i+2*ny,j+3*ha_wd]
        ph2[i+ha_wd,j]   = BC[i+2*ny,j+4*ha_wd]

        ps[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+0*ha_wd]
        ph[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+1*ha_wd]
        U[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+2*ha_wd]
        dpsi[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+3*ha_wd]
        ph2[i+ha_wd,j+ny+ha_wd] = BC[i+2*ny+nx,j+4*ha_wd]
   # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1]
    if i < ha_wd and j < ha_wd:
       ps[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+0*ha_wd];
       ph[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+1*ha_wd];
       U[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+2*ha_wd];
       dpsi[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+3*ha_wd];
       ph2[i,j]  =     BC[i+2*ny+2*nx+0*ha_wd,j+4*ha_wd];
       ps[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+0*ha_wd];
       ph[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+1*ha_wd];
       U[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+2*ha_wd];
       dpsi[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+3*ha_wd];
       ph2[i+nx+ha_wd,j] =   BC[i+2*ny+2*nx+1*ha_wd,j+4*ha_wd];

       ps[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+0*ha_wd];
       ph[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+1*ha_wd];
       U[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+2*ha_wd];
       dpsi[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+3*ha_wd];
       ph2[i,j+ny+ha_wd] =   BC[i+2*ny+2*nx+2*ha_wd,j+4*ha_wd];

       ps[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+0*ha_wd];
       ph[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+1*ha_wd];
       U[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+2*ha_wd];
       dpsi[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+3*ha_wd];
       ph2[i+nx+ha_wd,j+ny+ha_wd] = BC[i+2*ny+2*nx+3*ha_wd,j+4*ha_wd];

def BC_comm(sendbuf, recvbuf, nx ,ny,nt):
    
    ntag = 8*nt

    if ( px < nprocx-1 ):
           # right_send = phi_gpu[-2*ha_wd:-ha_wd,:].reshape((ha_wd,ny+2*ha_wd))
    #        print('rank',rank,'send data',time.time())
            comm.Send([sendbuf[ny:2*ny,:],MPI.DOUBLE], dest = rank+1, tag=ntag+1)

    if ( px > 0 ):
     #       print('rank',rank,'recv data',time.time())
            comm.Recv([recvbuf[:ny,:],MPI.DOUBLE],source = rank-1, tag=ntag+1)
           # phi_gpu[:ha_wd,:] = left_recv
    # sending direction:  X decreases
    if ( px > 0 ):
            comm.Send([sendbuf[:ny,:],MPI.DOUBLE], dest = rank-1, tag=ntag+2)
    if ( px < nprocx-1 ):
            comm.Recv([recvbuf[ny:2*ny,:],MPI.DOUBLE],source = rank+1, tag=ntag+2)

    # sending direction:  Y(Z) increases
    if ( py < nprocy-1 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+nprocx, tag=ntag+3)
    if ( py>0 ):
            comm.Recv([recvbuf[2*ny:2*ny+nx,:],MPI.DOUBLE],source = rank-nprocx, tag=ntag+3)
    # sending direction:  Y(Z) decreases
    if ( py >0 ):
            comm.Send([sendbuf[2*ny:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx, tag=ntag+4)
    if ( py < nprocy -1 ):
            comm.Recv([recvbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE],source = rank +nprocx, tag=ntag+4)


        # send and recv corner data, the order of send is [-1,-1],[0,-1],[-1,0],[0,0]
        # the order: [0,:],[-1,:],[:,0],[:,-1]
        # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1] send the corner data from [:,0] [:,-1]
    if ( px < nprocx-1 and py < nprocy-1):
            comm.Send([sendbuf[2*ny+2*nx-ha_wd:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+1+nprocx, tag=ntag+5)

    if ( px > 0 and py > 0 ):
            comm.Recv([recvbuf[-4*ha_wd:-3*ha_wd,:],MPI.DOUBLE],source = rank-1-nprocx, tag=ntag+5)
    # sending direction:  X decreases    send [0,0] recv [-1,-1]
    if ( px > 0 and py > 0):
            comm.Send([sendbuf[2*ny:2*ny+ha_wd,:],MPI.DOUBLE], dest = rank-1-nprocx, tag=ntag+6)
    if ( px < nprocx-1 and py < nprocy-1 ):
            comm.Recv([recvbuf[-ha_wd:,:],MPI.DOUBLE],source = rank+1+nprocx, tag=ntag+6)

    # sending direction:  Y(Z) increases  send [0,-1] recv [-1,0]
    if ( py < nprocy-1 and px > 0 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+nx+ha_wd,:],MPI.DOUBLE], dest = rank+nprocx-1, tag=ntag+7)
    if ( py>0 and px < nprocx-1 ):
            comm.Recv([recvbuf[-3*ha_wd:-2*ha_wd,:],MPI.DOUBLE],source = rank-nprocx+1, tag=ntag+7)
    # sending direction:  Y(Z) decreases  send [-1,0] recv [0,-1]
    if ( py>0 and px < nprocx-1):
            comm.Send([sendbuf[2*ny+nx-ha_wd:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx+1, tag=ntag+8)
    if ( py < nprocy -1 and px > 0):
            comm.Recv([recvbuf[-2*ha_wd:-ha_wd,:],MPI.DOUBLE],source = rank +nprocx-1, tag=ntag+8)

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
lminx = px*len_blockx + xmin_mic
lminz = py*len_blockz + zmin_mic

#dx = 1.2
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

filename = filename + 'hawd' + str(ha_wd)

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
theta0 = np.zeros((nv,nts+1), dtype=np.float64)
zz_mv = np.zeros((nz,nts+1), dtype=np.float64)
t_snapshot = np.zeros(nts+1, dtype=np.float64)
#Temp = np.zeros((nv,nts+1), dtype=np.float64)


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
    nx,nz=phi.shape
    while True :

        sum_arr[0] = 0.
        sum_cur_tip[bpg,tpb](sum_arr,  cur_tip, phi)
        mean_along_z = sum_arr[0] / nx

        if (mean_along_z > l2s) and cur_tip < nz-1:
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
       # if 1-ph[i,j]**2>0.01: alpha = alpha_m[i,j]
       # else: alpha = 0
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

        # a new section here to change the misorientation angle for boundary points
        if ph_new[i,j] > l2s and ph[i,j] < l2s: # flip the misorientation angle from 0 to angle of the nearest solid grid
          if -1e-15 < alpha < 1e-15:
             alphan = alpha; flip = False
             n1 = 0; n2 = 0; n3 = 0;
             alpha1 = 0; alpha2 = 0; alpha3 = 0;
             num_nbr = 9; len_nbr = 3;
             ## count the numbers for every angle:
             for iid in range(num_nbr):
                 xi = iid%len_nbr; yi = int(iid/len_nbr)
                 if ph[i+xi-1,j+yi-1] > l2s: 
                       temp = alpha_m[i+xi-1,j+yi-1]
                       if   alpha1==0: n1 +=1; alpha1 = temp;
                       elif -1e-15 < temp-alpha1 < 1e-15: n1 +=1
                       elif alpha2==0: n2 +=1; alpha2 = temp;
                       elif -1e-15 < temp-alpha2 < 1e-15: n2 +=1
                       elif alpha3==0: n3 +=1; alpha3 = temp;
                       elif -1e-15 < temp-alpha3 < 1e-15: n3 +=1 
                       else: print("case not closed!!!!")
             ### find the maximum number of occurence:
             if n1>=n2 and n1 >=n3: alpha_m[i,j] = alpha1
             elif n2>=n3: alpha_m[i,j] = alpha2
             else: alpha_m[i,j] = alpha3    
             if n1==0 and n2==0 and n3==0 and ha_wd-1<i<m-ha_wd and ha_wd-1<j<n-ha_wd:
                     print('psi',ps[i,j],ps_new[i,j],'i',i,'j',j,'cannot find the solid points in the 8 neighbors')
            # else: alpha_m[i,j] = alphan
         # else:
         #    if rank==0:
         #       print('alpha',alpha,'i',i,'j',j,'the angle of liquid should be set to zero initially!!!')
                #sys.exit(1)
        if ph_new[i,j] < l2s and ph[i,j] > l2s: # flip the misorientation angle from solid to zero
           alpha_m[i,j] = 0.0

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
def XYT_lin_interp(x, y, t, X, Y, T,  u_3d,u_m,v_2d,v_m ):
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

#        v_m[i,j] =  (1-delta_x)*(1-delta_y)*v_2d[kx,ky] + (1-delta_x)*delta_y*v_2d[kx,ky+1] \
#                     +delta_x*(1-delta_y)*v_2d[kx+1,ky] +   delta_x*delta_y*v_2d[kx+1,ky+1] 

    return

@cuda.jit
def rotate( Bid, len_box, nx0, nz0, alpha_arr, phiw, Uw, Tw, phib, Ub, Tb ):
    # this function gonna call every time step (in a region), define a space map
    # rotate the base to make the dendrites vertical (the base point B keeps at the same location).
    # then select a window of specified size nwx (odd number), nwz(odd number) around the base point
       cent = int((len_box-1)/2)
       alphaB = alpha_arr[Bid] 
       cosB = math.cos(alphaB); sinB = math.sin(alphaB)

       i,j = cuda.grid(2)

       nwx, nwz = phiw.shape

       if  i < nwx and j < nwz :   ## 0 <= i,j <=len_box-1, the middle int((len_box-1)/2)
 
         xij = cosB*(i-cent) - sinB*(j-cent)
         zij = sinB*(i-cent) + cosB*(j-cent)

         kx = nx0 + int(xij)
         delta_x = xij - int(xij)
         kz = nz0 + int(zij)
         delta_z = zij - int(zij)

         phiw[i,j] =  (1-delta_x)*(1-delta_z)*phib[kx,kz] + (1-delta_x)*delta_z*phib[kx,kz+1] \
                        +delta_x*(1-delta_z)*phib[kx+1,kz] +   delta_x*delta_z*phib[kx+1,kz+1]
 
         Uw[i,j] =  (1-delta_x)*(1-delta_z)*Ub[kx,kz] + (1-delta_x)*delta_z*Ub[kx,kz+1] \
                      +delta_x*(1-delta_z)*Ub[kx+1,kz] +   delta_x*delta_z*Ub[kx+1,kz+1]
         Tw[i,j] = (1-delta_x)*(1-delta_z)*Tb[kx,kz] + (1-delta_x)*delta_z*Tb[kx,kz+1] \
                      +delta_x*(1-delta_z)*Tb[kx+1,kz] +   delta_x*delta_z*Tb[kx+1,kz+1]
 
       return

# need to define a flag to close the QoI calculations. 
# before that flag turn to a false, keep the tracker open
def box_generator(x_1d, z_1d, num_boxx, num_boxz, Len, X, Z, alpha_micro):
    # CPU function call initially
    # leave margin in every GPU
    # dimensions x,y (nx+2*ha_wd, nz+2*ha_wd); 
    # define a tracker based on the location of base points (B), phi[B] > QoI_thre, start the rotation and calculate tip position
    # if the tip position out of the rotated box, turn the flag to false, and pass to cpu.
    # define another tracker phi[B] < vel_thre, between 2 trackers, record tip position every time step for tip velocity    
    
    xB = []; zB = [];       

    # 1. downsampling the macrodata to select the points B. 
    x_margin = 0.75*Len*dx; z_margin = 0.75*Len*dz
    xmin = x_1d[0] + x_margin;  xmax = x_1d[-1] - x_margin 
    zmin = z_1d[0] + z_margin;  zmax = z_1d[-1] - z_margin

    
   # x_in=(X>xmin)*1*(X<xmax); z_in=(Z>zmin)*1*(Z<zmax);
    xBid = [i for i, x in enumerate( (X>xmin)*1*(X<xmax) ) if x]; down_samx = int( len(xBid)/num_boxx ) 
    zBid = [i for i, x in enumerate( (Z>zmin)*1*(Z<zmax) ) if x]; down_samz = int( len(zBid)/num_boxz )
    xBid = xBid[::down_samx]; zBid = zBid[::down_samz]

    XX, ZZ = np.meshgrid(X,Z,indexing='ij')
    R = np.sqrt( (XX)**2 + (ZZ-center)**2)
    phi_macro = np.tanh((R-r0)/sqrt2)
    for i in xBid: 
     for j in zBid:
       if phi_macro[i,j] < l2s: 
          xB.append(XX[i,j]); zB.append(ZZ[i,j]); 
    
    num_box = len(xB); alphaB=np.zeros(num_box)
    xBid = np.zeros(num_box, dtype=int); zBid = np.zeros(num_box, dtype=int);
    for i in range(num_box):
          xBid[i] = np.argmin(np.absolute(x_1d-xB[i]));
          zBid[i] = np.argmin(np.absolute(z_1d-zB[i]));
          alphaB[i] = alpha_micro[xBid[i],zBid[i]]
    return num_box, xBid, zBid, alphaB 


def save_data(psi,U,misor,z):
    
    phi=np.tanh(psi)
    cinf_cl0 =  1+ (1-k)*U_0
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 ) / cinf_cl0
    misor = ((misor*180/pi)+90)*(phi>l2s)    
#    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )
   
    return np.reshape(psi[ha_wd:-ha_wd,ha_wd:-ha_wd],     (nv,1), order='F') , \
           np.reshape(c_tilde[ha_wd:-ha_wd,ha_wd:-ha_wd], (nv,1), order='F') , \
           np.reshape(misor[ha_wd:-ha_wd,ha_wd:-ha_wd],     (nv,1), order='F') , \
           z[ha_wd:-ha_wd,].T



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


elif ictype == 5: # radial initial condition

     # load U 1d data and construct radial for U and psi 
#     intp_data =
#   if len(sys.argv)==4:
     dd = sio.loadmat(sys.argv[2]) 
     points = dd['points']
     #psi_value = dd['psi_value']
     U_value = dd['U_value']
     r0 = dd['Rmax']/W0
     center = dd['cent']/W0
     r = np.sqrt( (xx)**2 + (zz-center)**2)
     psi0 =  r-r0
     #psi0 =  griddata(points, psi_value, (xx*W0,zz*W0), method='linear')[:,:,0]/W0

     phi0 = np.tanh(psi0/sqrt2)

     U0 =  griddata(points, U_value, (xx*W0,zz*W0), method='linear')[:,:,0] 

     n_theta = 120
   #  np.random.seed(seed_val)
  #   theta_arr = np.random.rand(n_theta)*pi/2
   #  print(theta_arr)
  #   theta = np.arctan(zz/xx)
  #   where_are_NaNs = np.isnan(theta)
  #   theta[where_are_NaNs] = 0
  #   i_theta = (np.absolute(theta/(pi/2/n_theta))).astype(int)

   #  alpha0=theta_arr[i_theta-1]*(phi0>l2s) 

     theta_arr = np.linspace(-pi/2,0,n_theta) 
     mac_data = sio.loadmat(sys.argv[2],squeeze_me = True)
     alpha_macro = -mac_data['n_alpha']
     X_cpu = mac_data['x_grid']/W0                # X_gpu , xx should be non-dimensional
     Z_cpu = mac_data['y_grid']/W0                # Z_gpu, zz, z_cpu should all be non-dimensional
     ainterp = interp2d(X_cpu, Z_cpu, alpha_macro.T,kind='cubic')
     theta = ainterp(x_1d ,z_1d)
     theta = theta.T 
     where_are_NaNs = np.isnan(theta)
     theta[where_are_NaNs] = 0
     i_theta = (np.absolute( (pi/2+theta) /(pi/2/n_theta))).astype(int)

     alpha0=theta_arr[i_theta-1]*(phi0>l2s)
     #print('i_theta', i_theta)
     #print('alpha0', alpha0)
     #generate QoI boxes:
     len_box = qoi_winds;
     cent = int((len_box-1)/2)
     box_per_gpu = 8
     R_max = 6000
     delta_box = len_box*(dx*W0)
     Mt_box= delta_box/R_max/(tau0*dt)
     interq = int(Mt_box/10)
     interq = interq if interq%2==0 else interq-1
     if rank==0: print('length',delta_box,'the shortest time step to pass the box', Mt_box, 'time interval', interq)  
     num_box, xB, zB, alphaB = box_generator(x_1d, z_1d, box_per_gpu, box_per_gpu, len_box, X_cpu, Z_cpu, theta)  
     print('length of box (grid points)',len_box,'number of box',num_box)
#     print('rank',rank,'the center of QoI boxes', num_box, xB, zB, alphaB)
 
else: 
    print('ERROR: invalid type of initial condtion...' )
    sys.exit(1)


# append halo around data
psi = set_halo(psi0)
phi = set_halo(phi0)
U = set_halo(U0)
alpha_cpu = set_halo(alpha0)
zz = set_halo(zz)
#print(z_1d)
x_cpu = np.zeros(nx+2*ha_wd); x_cpu[ha_wd:-ha_wd] = x_1d
z_cpu = np.zeros(nz+2*ha_wd); z_cpu[ha_wd:-ha_wd] = z_1d
#z_cpu  = zz[ha_wd,:]
#x_cpu = set_halo(xx)[:,ha_wd]       # add x array here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# set BCs
setBC_cpu(psi, 1, 1)
setBC_cpu(U, 1, 1)
setBC_cpu(phi,1,1)
setBC_cpu(alpha_cpu, 1 ,1)

phi_cpu = phi.astype(np.float64)
U_cpu = U.astype(np.float64)
psi_cpu = psi.astype(np.float64)
alpha_cpu = alpha_cpu.astype(np.float64)
print('the initial misorientations for liquid are all zeros', ( -1e-15< np.all( alpha_cpu*(phi<l2s)) <1e-15 ) )
# save initial data
op_phi[:,[0]], conc[:,[0]], theta0[:,[0]], zz_mv[:,0] = save_data(phi_cpu, U_cpu, alpha_cpu, z_cpu )




# allocate space on device
psi_old = cuda.to_device(psi_cpu)
phi_old = cuda.to_device(phi_cpu)
U_old   = cuda.to_device(U_cpu)

psi_new = cuda.device_array_like(psi_old)
phi_new = cuda.device_array_like(phi_old)
U_new = cuda.device_array_like(U_old)

T_m = cuda.device_array_like(U_old)
alpha_m = cuda.to_device(alpha_cpu)

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
X_gpu = cuda.to_device(mac_data['x_grid']/W0)     # X_gpu , xx should be non-dimensional
Z_gpu = cuda.to_device(mac_data['y_grid']/W0)     # Z_gpu, zz, z_cpu should all be non-dimensional
mac_t_gpu = cuda.to_device(mac_data['t_macro'])   # time here should have dimension second
T_3D_gpu = cuda.to_device(mac_data['T_arr'])

alpha_3D_cpu =-mac_data['n_alpha']
#alpha_3D_cpu = np.zeros_like(phi_cpu) +15/180*pi
alpha_3D_gpu = cuda.to_device(alpha_3D_cpu)

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
# set the arrays for QoIs

inter_len = np.zeros(num_box); pri_spac = np.zeros(num_box); sec_spac = np.zeros(num_box);
fs_arr = np.zeros((len_box,num_box)); cqois = np.zeros((10,num_box));
HCS = np.zeros(num_box);Kc_ave = np.zeros(num_box)
Ttip_arr = np.zeros(num_box);
ztip_qoi = np.zeros(num_box)
time_qoi = np.zeros(num_box)
tip_vel = np.zeros(num_box)

#### allocate the memory on GPU for QoIs calculation
phiw = cuda.device_array([len_box,len_box],dtype=np.float64) 
Uw   = cuda.device_array([len_box,len_box],dtype=np.float64)
Tw   = cuda.device_array([len_box,len_box],dtype=np.float64) 
xB_gpu = cuda.to_device(xB); zB_gpu = cuda.to_device(zB)
alphaB_gpu = cuda.to_device(alphaB); 
cp_cpu_flag = cuda.device_array(num_box,dtype=np.int32)
if num_box<100: num_frame = 2*len_box 
else: num_frame = len_box
tip_tracker_gpu = cuda.device_array([num_box,num_frame],dtype=np.int32) 
tip_count = cuda.device_array(num_box,dtype=np.int32)
tipB = cuda.device_array(num_box,dtype=np.int32)
print_flag = True; end_qoi_flag = False


# allocate boundary data
BCsend = cuda.device_array([2*nx+2*nz,5*ha_wd],dtype=np.float64);
BCrecv = cuda.device_array([2*nx+2*nz+4*ha_wd,5*ha_wd],dtype=np.float64)
#bpgBC = (2*bpg_x+2*bpg_y,1)
bpgBC = (2*(bpg_x+bpg_y),math.ceil(5*ha_wd/tpb) )
BC_421[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, alpha_m, BCsend)
#if rank == 0: print('sendbuf',BCsend.copy_to_host())
comm.Barrier()
BC_comm(BCsend, BCrecv, nx ,nz,0)
comm.Barrier()
BC_124[bpgBC,tpb2d](psi_old,phi_old,U_old, dPSI, alpha_m, BCrecv)
BC_124[bpgBC,tpb2d](psi_new,phi_new,U_new, dPSI, alpha_m, BCrecv)
#if rank == 0: print('recvbuf',BCrecv.copy_to_host())
setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, alpha_m,px, py, nprocx, nprocy, ha_wd)
setNBC_gpu[bpg,tpb](psi_new,phi_new,U_old,dPSI, alpha_m,px, py, nprocx, nprocy, ha_wd)
# march two steps per loop
start = time.time()
#print(U_cpu[ha_wd:-ha_wd,ha_wd:-ha_wd])
for kt in range(int(Mt/2)):
    if math.isnan(psi_old[-ha_wd-1,-ha_wd-1])==True: 
        print('rank',rank,'blow up')
        #psi=psi_old.copy_to_host()
        #print(psi[ha_wd:-ha_wd,ha_wd:-ha_wd]);#sys.exit(1)
    # =================================================================
    # time step: t = (2*nt) * dt
    # =================================================================
    t_cur = (2*kt)*dt*tau0
    XYT_lin_interp[bpg2d, tpb2d](x_gpu, z_gpu, t_cur, X_gpu, Z_gpu, mac_t_gpu, T_3D_gpu, T_m, alpha_3D_gpu, alpha_m )
    rhs_psi[bpg2d, tpb2d](psi_old, phi_old, U_old, psi_new, phi_new, U_new, z_gpu, dPSI, 2*kt, rng_states, T_m, alpha_m)
    #if ha_wd==1:
    if (2*kt+2)%ha_wd==0:
      BC_421[bpgBC,tpb2d](psi_new, phi_new, U_old, dPSI,alpha_m, BCsend)
      comm.Barrier()
    #  print('ready to send data', rank , time.time())
      BC_comm(BCsend, BCrecv, nx ,nz,2*kt+1)
      comm.Barrier()
    #  print('finish receive data', rank, time.time())
      BC_124[bpgBC,tpb2d](psi_new, phi_new, U_old, dPSI,alpha_m, BCrecv)
    setNBC_gpu[bpg,tpb](psi_new,phi_new,U_old,dPSI, alpha_m, px, py, nprocx, nprocy, ha_wd)
    rhs_U[bpg2d, tpb2d](U_old, U_new, phi_new, dPSI)
    # =================================================================
    # time step: t = (2*nt+1) * dt
    # =================================================================
    t_cur = (2*kt+1)*dt*tau0 
    XYT_lin_interp[bpg2d, tpb2d](x_gpu, z_gpu, t_cur, X_gpu, Z_gpu, mac_t_gpu, T_3D_gpu, T_m, alpha_3D_gpu, alpha_m )
    rhs_psi[bpg2d, tpb2d](psi_new, phi_new, U_new, psi_old, phi_old, U_old, z_gpu, dPSI, 2*kt+1, rng_states, T_m, alpha_m)
 
    if (2*kt+2)%ha_wd==0: #ha_wd==1 or ha_wd==2:
      BC_421[bpgBC,tpb2d](psi_old,phi_old,U_new, dPSI, alpha_m,BCsend)
      comm.Barrier()
      BC_comm(BCsend, BCrecv, nx ,nz,2*kt+2)
      comm.Barrier()
      BC_124[bpgBC,tpb2d](psi_old,phi_old,U_new, dPSI, alpha_m, BCrecv)
    setNBC_gpu[bpg,tpb](psi_old,phi_old,U_new,dPSI, alpha_m, px, py, nprocx, nprocy, ha_wd)
    rhs_U[bpg2d, tpb2d](U_new, U_old, phi_old, dPSI) 

    ## QoI section: the windows are fixed at the beginning, the choice we have is when to calculate QoIs.
    ## time_st_qoi: necessary, start the rotate--cal_tip, criterion: phiB > -0.999
    ## time_vel: necessary, end the list for tip position, criterion: cur_tip > center + 5
    ## time_end_qoi: this is not determined yet, for now criterion: cur_tip> max-5 need to transfer phi, U to cpu 
    if (2*kt+2)%interq == 0 and U_old[ha_wd,ha_wd] > l2s and end_qoi_flag == False :
       for Bid in range(num_box):
         nx0 = xB_gpu[Bid] + ha_wd; nz0 = zB_gpu[Bid] + ha_wd;
         if cp_cpu_flag[Bid] ==0:  ## assume dendrites grow in z direction, start the tip tracker
            #if phi_old[nx0-10,nz0-10]>l2s or phi_old[nx0,nz0-10]>l2s or phi_old[nx0-10,nz0]>l2s: 
           if U_old[nx0-cent,nz0-cent]> l2s : #or phi_old[nx0,nz0-cent]>l2s or phi_old[nx0-cent,nz0]>l2s:  
             #print('rank',rank,'box id', Bid)
              rotate[bpg2d, tpb2d]( Bid, len_box, nx0, nz0, alphaB_gpu, phiw, Uw, Tw, phi_old, U_old, T_m )
               
             # print('the tip poisition stored right now',tipB[Bid]) 
              cur_tip_x, cur_tip= compute_tip_pos(tipB[Bid], sum_arr, phiw) 
              tipB[Bid] = cur_tip  
              if tip_count[Bid] < num_frame:
                 if tip_count[Bid]==0 and cur_tip>cent: print('got tip position larger than the center initially !!!')
                 if tip_count[Bid]==1 and cur_tip==tip_tracker_gpu[Bid,0]: tip_count[Bid]=1       
                 else: tip_tracker_gpu[Bid,tip_count[Bid]] = cur_tip; tip_count[Bid] +=1; \
                       print('the current tip position ', cur_tip, ' in the box no.', Bid, 'rank', rank)
              if cur_tip>len_box-5: 
                 print('the box no.', Bid, 'in rank',rank,' turn off and transfer data to cpu, current tip', cur_tip )
                 phi_cp = phiw.copy_to_host().T
                 U_cp  = Uw.copy_to_host().T
                 T_cp = Tw.copy_to_host().T
                 tip_cp = tip_tracker_gpu[Bid,:].copy_to_host()
                 c_cp = ( 1+ (1-k)*U_cp )*( k*(1+phi_cp)/2 + (1-phi_cp)/2 ) / ( 1+ (1-k)*U_0 )
                 ## and the relavent QoI calculations
                 cp_cpu_flag[Bid] =1
                 inter_len[Bid] = interf_len(phi_cp,W0)
                 pri_spac[Bid], sec_spac[Bid] = spacings(phi_cp, cur_tip, (len_box-1)*dx*W0, dxd, mph)
                 tip_cp = tip_cp[tip_cp>0.5]; vel_arr = np.diff(tip_cp)*dx*W0/(interq*dt*tau0);vel_itp = interp1d(tip_cp[:-1],vel_arr) 
                 tip_vel[Bid] = vel_itp(cent);#print('velocity distribution',vel_arr)
                 cqois[:,Bid] = conc_var(phi_cp,c_cp) 
                 Tz_cp = np.mean(T_cp, axis=1)
                 fs_arr[:, Bid] = solid_frac(phi_cp,  821, Tz_cp)
                 fs_cur = smooth_fs( fs_arr[:,Bid], len_box-2 )
                 bool_arr= (fs_cur>1e-2)*(fs_cur<1)
                 fs_cur = fs_cur[bool_arr]; Tz_cp = Tz_cp[bool_arr]
                 HCS[Bid], HCS_arr = Kou_HCS(fs_cur, Tz_cp)
                 Kc_ave[Bid] = np.mean( permeability(fs_cur,pri_spac[Bid], mph) )
                 

    if sum(cp_cpu_flag)==num_box and print_flag==True: 
           end_qoi_flag = True; print_flag = False; print('rank',rank,'ends QoI section!!!!!!!')

    if (2*kt+2)%kts==0:
#       
       print('time step = ', 2*(kt+1) )
    #   if mvf == True: print('tip position nz = ', cur_tip)
       kk = int(np.floor((2*kt+2)/kts))
       phi = psi_old.copy_to_host()
       U  = U_old.copy_to_host()
       misor = alpha_m.copy_to_host()
     #  temp = T_m.copy_to_host()
       z_cpu = z_gpu.copy_to_host()
       op_phi[:,[kk]], conc[:,[kk]], theta0[:,[kk]], zz_mv[:,kk] = save_data(phi,U, misor, z_cpu) 

       t_snapshot[kk] = 2*(kt+1)*dt

tip_boxes = tip_tracker_gpu.copy_to_host()

end = time.time()
print('elapsed time: ', (end-start))

if num_box!=0: 
  save(os.path.join(direc,filename+'.mat'),{'op_phi':op_phi, 'conc':conc, 'theta0':theta0, 'x':x_1d*W0, 'z':z_1d*W0,'dt':dt*tau0,\
  'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start,'t_snapshot':t_snapshot*tau0,'xB':x_1d[xB],'zB':z_1d[zB],'alphaB':alphaB,\
  'num_box':num_box,'phi_win':phi_cp,'c_win':c_cp,'T_win':T_cp,'tip_boxes':tip_boxes,'interf_len':inter_len,'pri_spac':pri_spac,'sec_spac':sec_spac,'HCS':HCS,\
'Kc_ave':Kc_ave,'cqois':cqois,'tip_vel':tip_vel} )
else:
  save(os.path.join(direc,filename+'.mat'),{'op_phi':op_phi, 'conc':conc, 'theta0':theta0, 'x':x_1d*W0, 'z':z_1d*W0,'dt':dt*tau0,\
  'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start,'t_snapshot':t_snapshot*tau0,'num_box':num_box} )
# 'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start, 't_snapshot':t_snapshot*tau0, 'temp':Temp[ha_wd:-ha_wd,ha_wd:-ha_wd], 'a_field':a_field[ha_wd:-ha_wd,ha_wd:-ha_wd]} )

'''
save(os.path.join(direc,filename+'_QoIs.mat'),{'time_qoi':time_qoi, 'ztip_qoi':ztip_qoi-ztip_qoi[0],\
'Ttip_arr':Ttip_arr,'tip_uq':tip_uq,'cqois':cqois,'pri_spac':pri_spac,'sec_spac':sec_spac,'interfl':inter_len,\
'fs_arr':fs_arr,'HCS':HCS,'Kc_ave':Kc_ave})

# save('initial.mat',{'phi_ic':phi, 'U_ic':U, 'psi_ic':psi, 'tip_x':cur_tip_x, 'tip_z':cur_tip, 'zz_mv':zz_mv})

save(os.path.join(direc,filename),{'order_param':order_param, 'conc':conc, 'xx':xx*W0, 'zz_mv':zz_mv*W0,'dt':dt*tau0,\
 'nx':nx,'nz':nz,'Tend':(Mt*dt)*tau0,'walltime':end-start,'ztip':ztip_arr,'Tip':Ttip_arr,'inter_len':inter_len,'pri_spac':pri_spac,\
    'sec_spac':sec_spac,'alpha':alpha_arr,'fs':fs_arr } )
'''
