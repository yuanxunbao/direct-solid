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
from numba import njit, stencil, vectorize, float32, float64
import numpy as np
import math
from numpy.random import random
import time

PARA = importlib.import_module(sys.argv[1])
#import dsinput as PARA

delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_infty, G, R = PARA.phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, seed, U_0, nts, filename, direc, \
mvf, Nset, ictype  = PARA.simu_para(W0,Dl_tilde)

k=1
alpha0 = alpha0*math.pi/180

cosa = np.cos(alpha0)
sina = np.sin(alpha0)

a_s = 1 - 3*delta
epsilon = 4*delta/a_s
a_12 = 4*a_s*epsilon
sqrt2 = np.sqrt(2.)

lx = lxd/W0
lz = aratio*lx

nz = int(aratio*nx+1)
nx = nx+1
nv= nz*nx #number of variables
dx = lx/(nx-1)
dz = lz/(nz-1)
print("mesh dx,dz",dx,dz)

x = np.linspace(0,lx,nx)
z = np.linspace(0,lz,nz)

zz,xx = np.meshgrid(z,x)
t=0

dxdz_in = 1./(dx*dz)  
dxdz_in_sqrt = np.sqrt(dxdz_in)

hi= 1./dx
dt_sr = np.sqrt(dt)


order_param = np.zeros((nv,nts+1))
conc = np.zeros((nv,nts+1))

np.random.seed(seed)

#Nset = nz - 81 #int(7/8*nz)
Ntip=1
Ntip_arr= np.zeros(Mt)
ztip_arr= np.zeros(Mt)

def move_frame(Ntip, psi, phi, U, zz):
    
    
    #phi_tip = phi[Ntip,:]
    phi_tip = phi[1:-1,Ntip]
    
    while np.mean(phi_tip)>-0.99:
        
        Ntip += 1
        
        #phi_tip = phi[Ntip,:]
        phi_tip = phi[1:-1,Ntip]
        
        
    if Ntip==Nset:
        # move everything down
        #flag = 1
        Ntip = Ntip -1
        U_new = U_0*np.ones((nx,1))
        zz_new = 2*zz[1:-1,[-2]] - zz[1:-1,[-3]]
        psi_new = 2*psi[1:-1,[-2]] - psi[1:-1,[-3]]
        
        U[1:-1,1:-1] = np.hstack( (U[1:-1,2:-1], U_new ))
        zz[1:-1,1:-1] = np.hstack(( zz[1:-1,2:-1], zz_new ))
        psi[1:-1,1:-1] = np.hstack(( psi[1:-1,2:-1], psi_new ))
   # else: 
        #flag =0
    
    return  psi, U, zz, Ntip, zz[3,Ntip]


@njit
def set_halo(u):
    
    m,n = u.shape
    ub = np.zeros((m+2,n+2))
    
    ub[1:-1,1:-1] = u
    
    return ub



@vectorize([float32(float32, float32),
            float64(float64, float64)])
def atheta(ux, uz):

    ux2 = ( cosa*ux + sina*uz )**2
    uz2 = ( -sina*ux + cosa*uz)**2
        
    # return MAG_sq2
    MAG_sq  = ux2 + uz2
    MAG_sq2 = MAG_sq**2
    
    if (MAG_sq > eps):
        
        return a_s*( 1 + epsilon*(ux2**2 + uz2**2) / MAG_sq2   )
        # return uz/MAG_sq2
    else:
        return 1.0
    
    
@vectorize([float32(float32, float32),
            float64(float64, float64)])
def aptheta(ux, uz):
    uxr = cosa*ux + sina*uz
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
        
        

@stencil
def _rhs_psi(ph,U,zz):

    # ps = psi, ph = phi
    
    # =============================================================
    # 
    # 1. ANISOTROPIC DIFFUSION
    # 
    # =============================================================
    
    # these ps's are defined on cell centers

    
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * 0.25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * 0.25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * 0.25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * 0.25
    
    # ============================
    # right edge flux
    # ============================

    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JR = A * ( A*phx - Ap*phz )
    
    # ============================
    # left edge flux
    # ============================

    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JL = A * ( A*phx - Ap*phz )
    
    # ============================
    # top edge flux
    # ============================

    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]


    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JT = A * ( A*phz + Ap*phx )
    
    # ============================
    # bottom edge flux
    # ============================

    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JB = A * ( A*phz + Ap*phx )
    
    
    
    # =============================================================
    # 
    # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
    # 
    # =============================================================
    
    # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)
    phxn = ( ph[ 1, 0] - ph[-1, 0] ) * 0.5
    phzn = ( ph[ 0, 1] - ph[ 0,-1] ) * 0.5

    
    A2 = atheta(phxn,phzn)**2
   # gradps2 = (psxn)**2 + (pszn)**2
    extra = 0 # -sqrt2 * A2 * ph[0,0] * gradps2
    

    # =============================================================
    # 
    # 3. double well (transformed): sqrt2 * phi + nonlinear terms
    # 
    # =============================================================
    
    Up = 0 #(zz[0,0] )/lT_tilde
    
    rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
               - lamd* (1-ph[0,0]**2)**2 *(U[0,0] + Up) + \
                   ph[0,0] - ph[0,0]**3
        
    
    # =============================================================
    # 
    # 4. dpsi/dt term
    # 
    # =============================================================
    tp = (1-(1-k)*Up)
    tau_psi = tp*A2 if tp >= k else k*A2

    # return rhs_psi/tau_psi + eta*(random()-0.5)/dt_sr*hi
    return rhs_psi/tau_psi



@stencil
def _rhs_U(U,ph,phi_t):
    
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
    
    jat    = (1+(1-k)*U[0,0])*phi_t[0,0]/sqrt2
    jat_ip = (1+(1-k)*U[1,0])*phi_t[1,0]/sqrt2
        
    UR = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[1,0])*(U[1,0]-U[0,0]) + \
         0.5*(jat + jat_ip)*nx
    UR = hi*Dl_tilde*  (U[1,0]-U[0,0])   
         
    # ============================
    # left edge flux (i-1/2, j)
    # ============================
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    phn2 = phx**2 + phz**2
    nx = phx / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_im = (1+(1-k)*U[-1,0])*phi_t[-1,0]/sqrt2
    
    UL = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[-1,0])*(U[0,0]-U[-1,0]) + \
         0.5*(jat + jat_im)*nx
    UL = hi*Dl_tilde*  (U[0,0]-U[-1,0])   
         
    # ============================
    # top edge flux (i, j+1/2)
    # ============================     
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
          
    jat_jp = (1+(1-k)*U[0,1])*phi_t[0,1]/sqrt2      
    
    UT = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[0,1])*(U[0,1]-U[0,0]) + \
         0.5*(jat + jat_jp)*nz
    UT = hi*Dl_tilde*  (U[0,1]-U[0,0])     
         
    # ============================
    # top edge flux (i, j-1/2)
    # ============================  
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_jm = (1+(1-k)*U[0,-1])*phi_t[0,-1]/sqrt2      
    
    UB = hi*Dl_tilde*0.5*(2 - ph[0,0] - ph[0,-1])*(U[0,0]-U[0,-1]) + \
         0.5*(jat + jat_jm)*nz
    UB = hi*Dl_tilde*  (U[0,0]-U[0,-1])
    rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat
    tau_U = (1+k) - (1-k)*ph[0,0]
    
    
    return rhs_U/tau_U
    

@njit(parallel=True)
def rhs_phi(ph,U,zz): return _rhs_psi(ph,U,zz)


@njit(parallel=True)
def rhs_U(U,ph,phi_t): return _rhs_U(U,ph,phi_t)



def save_data(phi,U):

    cr =  1+ (1-k)*U_0
    c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )/cr
    # c_tilde = ( 1+ (1-k)*U )*( k*(1+phi)/2 + (1-phi)/2 )

    return np.reshape(phi[1:-1,1:-1],     (nv,1), order='F') , \
           np.reshape(c_tilde[1:-1,1:-1], (nv,1), order='F')



if ictype == 0:

    psi0 = PARA.seed_initial(xx,lx,zz)
    phi0 = np.tanh(psi0/sqrt2)
    U0 = U_0*(phi0<=0)  ## for specific case in the paper

elif ictype == 1:

    psi0 = PARA.planar_initial(lz,zz)
    U0 = 0*psi0 + U_0
    phi0 = np.tanh(psi0/sqrt2)

elif ictype == 2:

    psi0 = PARA.sum_sine_initial(lx,nx,xx,zz)
    U0 = 0*psi0 + U_0
    phi0 = np.tanh(psi0/sqrt2)

else:
    print('ERROR: invalid type of initial condtion...' )
    sys.exit(1)

# append halo
psi = set_halo(psi0)
U = set_halo(U0)
zz = set_halo(zz)

psi = set_BC(psi, 1, 1)
phi = np.tanh(psi/sqrt2)   # expensive replace
U =   set_BC(U, 1, 1)
order_param[:,[0]], conc[:,[0]] = save_data(phi,U)


# For all numba routines to JIT-compile
start = time.time()
dPSI = rhs_phi(phi, U, zz)
dPSI = set_BC(dPSI, 1, 1)
dU = rhs_U(U,phi,dPSI)
end = time.time()

print('compile time: ', end - start )

print(psi.shape)

kts = int(Mt/nts)
start = time.time()

for ii in range(Mt):

    dPSI = rhs_phi(phi, U, zz - R_tilde*t)

    dPSI = set_BC(dPSI, 1, 1)
    
    beta = random(psi.shape) - 0.5
    
    phi = phi + dt*dPSI + dt_sr*dxdz_in_sqrt*beta*eta
    phi = set_BC(phi, 1, 1)
  
    U = U + dt*rhs_U(U,phi,dPSI)
    U = set_BC(U, 1, 1)    
    
    # =================================================================
    # If moving frame flag is set to TRUE
    # =================================================================
    if mvf == True :
    # check cross Nset, if true, move down
        psi, U, zz, Ntip,ztip = move_frame(Ntip, psi, phi, U, zz)
        Ntip_arr[ii]=Ntip; ztip_arr[ii]=ztip
    
    # add boundary
    #phi = set_BC(phi, 1, 1)
    #U = set_BC(U, 1, 1)
    
 
    # update phi
    #phi = np.tanh(psi/sqrt2) 
    t += dt
    
    
    if (ii+1)%kts==0:     # data saving 
       
        print('time step = ', (ii+1) )
        if mvf == True: print('tip position nz = ', Ntip)
         
        kk = int(np.floor((ii+1)/kts))
        order_param[:,[kk]], conc[:,[kk]] = save_data(phi,U)
    
end = time.time()


print('elapsed: ', end - start )


Uf = U[1:-1,1:-1]

save(os.path.join(direc,filename),{'xx':xx*W0,'zz':zz[1:-1,1:-1]*W0,'order_param':order_param,'conc':conc,'Ntip':Ntip_arr,'ztip':ztip_arr,'dt':dt*tau0,'nx':nx,'nz':nz,'Tend':t*tau0,'walltime':end-start,'dPSI':dPSI})
