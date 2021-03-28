#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:53 2020

@author: yigongqin
"""



import numpy as np
import math

def phys_para():    
# NOTE: for numbers entered here, if having units: length in micron, time in second, temperature in K.
    G = 0.02                        # thermal gradient        K/um
    R = 50                          # pulling speed           um/s
    delta = 0.02                    # strength of the surface tension anisotropy         
    k = 0.14                        # interface solute partition coefficient
    c_infm = 1.519                  # shift in melting temperature     K
    Dl = 3000                       # liquid diffusion coefficient      um**2/s
    d0 = 0.02572                    # capillary length -- associated with GT coefficient   um
    W0 = 0.9375                     # interface thickness      um
    
    lT = c_infm*( 1.0/k-1 )/G       # thermal length           um
    lamd = 5*np.sqrt(2)/8*W0/d0     # coupling constant
    tau0 = 0.6267*lamd*W0**2/Dl     # time scale               s
    
    c_infty = 2.45e-3
    # non-dimensionalized parameters based on W0 and tau0
    
    R_tilde = R*tau0/W0
    Dl_tilde = Dl*tau0/W0**2
    lT_tilde = lT/W0

    return delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, c_infty, G, R

def simu_para(W0,Dl_tilde):
    
    eps = 1e-8                      	# divide-by-zero treatment
    alpha0 = 0                    	# misorientation angle in degree
    
    accl = 1
    asp_ratio = 4                  	# aspect ratio
    nx = 128            		# number of grids in x   nx*aratio must be int
    lxd = 1.2*W0*nx                     # horizontal length in micron
    dx = lxd/nx/W0
    dt = 0.8*(dx)**2/(4*Dl_tilde)*accl       # time step size for forward euler
    Mt = int(10000/accl)                      	# total  number of time steps

    eta = 0.0                		# magnitude of noise

    seed_val = np.uint64(np.random.randint(1,1000))
    U0 = -0.3                		# initial value for U, -1 < U0 < 0
    nts = 10				# number snapshots to save, Mt/nts must be int
    mv_flag = False			# moving frame flag
    tip_thres = np.int32(math.ceil(0.7*nx*asp_ratio))
    ictype = 0                   	# initial condtion: 0 for semi-circular, 1 for planar interface, 2 for sum of sines
    filename = 'phi_code.mat'
    direc = './'                	# direc = '/scratch/07429/yxbao/data'    # saving directory
    # filename = 'dirsolid_cpu_noise' + str('%4.2E'%eta)+'_misori'+str(alpha0)+'_lx'+ str(lxd)+'_nx'+str(nx)+'_asp'+str(asp_ratio)+'_seed'+str(seed_val)+'.mat'
    
    
    

    return eps, alpha0, lxd, asp_ratio, nx, dt, Mt, eta, seed_val, U0, nts, filename,direc, mv_flag, tip_thres, \
           ictype

def seed_initial(xx,lx,zz): 
    
    r0 = 0.5625 * 10
    r = np.sqrt( (xx-lx/2) **2+(zz)**2)     
    psi0 = r0 - r 
    
    return psi0


def planar_initial(lz,zz):
    
    z0 = lz*0.4                   # initial location of interface in units of W0   
    psi0 = z0 - zz
    
    return psi0


def sum_sine_initial(lx,nx,xx,zz): 
    
    k_max = int(np.floor(nx/10))    # max wavenumber, 12 grid points to resolve the highest wavemode
    
    amp = 1
    A = (np.random.rand(k_max)-0.5) * amp  # amplitude, draw from [-1,1] * eta
    x_c = np.random.rand(k_max)*lx;                  # shift, draw from [0,Lx]    
    z0 = lx*0.01;                               # level of z-axis
    
    # sinusoidal perturbation
    sp = 0*zz
    for kk in range(k_max):
       
        sp = sp + A[kk]*np.sin(2*math.pi*kk/lx* (xx-x_c[kk]) );
        
    psi0 = -(zz-z0-sp)
    
    return psi0
