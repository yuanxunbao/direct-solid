#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:53 2020

@author: yigongqin
"""



import numpy as np
import math
import scipy.io as sio


def phys_para(macrodata):    
# NOTE: for numbers entered here, if having units: length in micron, time in second, temperature in K.

    macroGR = sio.loadmat(macrodata, squeeze_me=True)    

    G = macroGR['G_t'][0]           # thermal gradient        K/um
    print('G', G) 
    R = 0.                          # pulling speed           um/s
    delta = 0.01                    # strength of the surface tension anisotropy         
    k = 0.14                        # interface solute partition coefficient
    m = 2.6
    c_inf = 3
    c_infm = m * c_inf              # shift in melting temperature     K
    Dl = 3000                       # liquid diffusion coefficient      um**2/s
    d0 = 5.0e-3                     # capillary length -- associated with GT coefficient   um
    W0 = 0.1132*1.5                 # interface thickness      um
    
    lT = c_infm*( 1.0/k-1 )/G       # thermal length           um
    lamd = 5*np.sqrt(2)/8*W0/d0     # coupling constant
    tau0 = 0.6267*lamd*W0**2/Dl     # time scale               s
    
    print('lambda = ', lamd)
   
    Te = 821
    Tm = 933.3
    Ti = Tm- c_infm/k 
    # U_0 = ( c_infm/( Tm - Ti ) - 1 )/(1-k)
    
    U_0 = -1.
    # non-dimensionalized parameters based on W0 and tau0
    
    R_tilde = R*tau0/W0
    Dl_tilde = Dl*tau0/W0**2
    lT_tilde = lT/W0


    return delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_inf, m, G, R, Ti, U_0
  
def simu_para(W0,Dl_tilde):
    
    eps = 1e-4                      	# divide-by-zero treatment
    alpha0 = 0                    	# misorientation angle in degree
    
    
    asp_ratio = 100                  	# aspect ratio
	       		                # number of grids in x   nx*aratio must be int
    lx = 10.0/W0                         # horizontal length in units of W0
    dx = 1.2
    nx = np.floor(lx/dx)
    dt = 0.8*(dx)**2/(4*Dl_tilde)       # time step size for forward euler
    Mt = 300000                     	# total  number of time steps

    eta = 0.0                		# magnitude of noise

    seed_val = np.uint64(np.random.randint(1,1000))
    #U0 = -0.3                		# initial value for U, -1 < U0 < 0
    nts = 5				# number snapshots to save, Mt/nts must be int
    mv_flag = True			# moving frame flag
    tip_thres = np.int32(math.ceil(0.7*nx*asp_ratio))
    ictype = 1                 	# initial condtion: 0 for semi-circular, 1 for planar interface, 2 for sum of sines

    direc = '/scratch/07428/ygqin/data'                	# direc = '/scratch/07429/yxbao/data'    # saving directory
    # filename = 'dirsolid_gpu_noise' + str('%4.2E'%eta)+'_misori'+str(alpha0)+'_lx'+ str(lxd)+'_nx'+str(nx)+'_asp'+str(asp_ratio)+'_seed'+str(seed_val)+'.mat'
    qts = 2*nts
    
    

    return eps, alpha0, lx, asp_ratio, nx, dt, Mt, eta, seed_val, nts, direc, mv_flag, tip_thres, \
           ictype, qts

def seed_initial(xx,lx,zz): 
    
    r0 = 0.5625 * 10 
    r = np.sqrt( (xx-lx/2)**2 + zz**2)     
    psi0 = r0 - r 
    
    return psi0


def planar_initial(lz,zz,z0):
    
    z0 = lz*0.0                   # initial location of interface in W0   
    psi0 = z0 - zz
    
    return psi0



def sum_sine_initial(lx,nx,xx,zz,z0): 
    
    k_max = int(np.floor(nx/nx))    # max wavenumber, 12 grid points to resolve the highest wavemode
    
   
    np.random.seed(0)
    amp = 2.
    A = (np.random.rand(k_max)-0.5) * amp  # amplitude, draw from [-1,1] * eta
    A[0]= 2

    x_c = np.random.rand(k_max)*lx*0;                  # shift, draw from [0,Lx]    
    # z0 = lz*0.01;                               # level of z-axis
    
    # sinusoidal perturbation
    sp = 0*zz
    for kk in range(k_max):

        sp = sp + A[kk]*np.sin(2*math.pi*(kk+1)/lx* (xx-x_c[kk]) );
        
    psi0 = -(zz-z0-sp)
    
    return psi0
