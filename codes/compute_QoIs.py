#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:33:02 2020

@author: yigongqin
"""


import numpy as np
from dsinput_gpu import phys_para, simu_para
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
from QoIs import ztip_Ntip,phi_xstat,spacings,solid_frac,tcp,tcpa,smooth_fs, crosspeak_counter,\
    Kou_HCS, permeability,interf_len,eutectic_Vfrac,solute_variability,tilted_angle,plumb
#from scipy.signal import savgol_filter

delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0, G, R = phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, seed, U0, nts, filename, direc, \
    mvf, tip_thres, ictype = simu_para(W0,Dl_tilde)


Q = G*R   # cooling rate

c_infty = 2.45e-3

lzd = aratio*lxd

nz = int(aratio*nx+1)

nv= nz*nx #number of variables

dxd = lxd/nx
dT = G*dxd
#xd = np.linspace(0,lxd-dxd,nx)
#zd = np.linspace(0,lzd,nz)
#xxd, zzd = np.meshgrid(xd, zd)

Ttd = Mt*dt*tau0

Ti = 931.20
Te = 821

Ntip = 1

mph = 'dendrite'
#mph = 'cell'

## ======================= load data ==========================##
# loadfile1 = 'move_eta004.mat'
loadfile1 = 'dirsolid_gpu_noise4.00E-02_misori15_lx2812.5_nx2000_asp0.5_seed105'


op = (loadmat(loadfile1)['order_param'])
cc = (c_infty*loadmat(loadfile1)['conc'])

phi = op[:,-1].reshape((nx,nz),order='F').T
conc = cc[:,-1].reshape((nx,nz),order='F').T

zz = (loadmat(loadfile1)['zz']).T

Tz =  Ti + G*( zz[:,3] - R*Ttd )

'''
#plot load data
fig1 = plt.figure() 
ax11 = fig1.add_subplot(121)
plt.title('phi')
plt.imshow(phi,cmap=plt.get_cmap('winter'),origin='lower')
ax12 = fig1.add_subplot(122)
plt.title('U')
plt.imshow(conc,cmap=plt.get_cmap('winter'),origin='lower')
'''
# print data informattion
print('the morphology of solid: ', mph)
print('the dimensions of data: ', nx, '*', nz)
print('the range of z coordinates: ', zz[0,1],'um ~',zz[-1,3],'um')
T_bottom = Tz[0]; T_top = Tz[-1]
print('the range of temperature: ', T_bottom,'K ~',T_top,'K')
print('\n')


##==================== QoIs =======================##


if T_bottom<Te and Te<T_top:
    
    euV = eutectic_Vfrac(phi, 0, Tz, Te)
    sol_var = solute_variability(conc, 0, Tz, Te)
    print('eutectic volume fraction: ', euV)
    print('solute variability: ', sol_var)
        
else:
    euV = 0; sol_var = 0
    print('the eutectic temperature is out of range of current window')

# tip information
ztip, Ntip = ztip_Ntip( phi, zz, Ntip)
print('tip coordinate: ', Ntip)
print('tip z location: ', ztip, 'um')
Ttip = Ti + G*( ztip - R*Ttd )
print('tip temperature: ',Ttip ,'K')

# dendrite structure
interf_len = interf_len(phi)
print('length of interface: ', interf_len)
print('\n')

if alpha0==0:
    print('the misorinetation angle is 0')
    mui, sigmai, var_phi = phi_xstat(phi,Ntip)
    #plt.plot(sigmai) #plot and check if you get the right number of cells
    print('variation of \phi: ', var_phi)
    pri_spac, sec_spac = spacings(phi, Ntip, lxd, dxd, mph)
    print('primary spacing is: ',pri_spac, 'um')
    print('secondary spacing is: ',sec_spac, 'um')
    
else:
    imid = int(nz/2)
    num_cell, est_len = crosspeak_counter(phi[Ntip-50,:], dxd, 0)
    est_pri = lxd/num_cell
    #print(est_pri) estimated primary spacing
    Npri = round(est_pri/dxd)
    alpha = tilted_angle(phi[imid:imid+10,:], phi[0:10,:], imid, Npri, int(np.sign(alpha0)))
    print('the measured growth angle is: ', alpha)
    phi = plumb(phi,alpha); conc = plumb(conc,alpha)
    '''
    #plot rotated data
    fig11 = plt.figure() 
    ax111 = fig11.add_subplot(121)
    plt.title('phi')
    plt.imshow(phi,cmap=plt.get_cmap('winter'),origin='lower')
    ax112 = fig11.add_subplot(122)
    plt.title('U')
    plt.imshow(conc,cmap=plt.get_cmap('winter'),origin='lower')
    '''
    mui, sigmai, var_phi = phi_xstat(phi,Ntip)
    plt.plot(mui) #plot and check if you get the right number of cells
    print('variation of \phi: ', var_phi)
    pri_spac, sec_spac = spacings(phi, Ntip, lxd, dxd, mph)
    print('primary spacing is: ',pri_spac, 'um')
    print('secondary spacing is: ',sec_spac, 'um')    
    
phi_cp = tcp(phi,Ntip,-5); Tz_cp = tcpa(Tz,Ntip,-5)
fs = solid_frac(phi_cp, Ntip, Te, Tz_cp)
#fs = smooth_fs(fs,11)
HCS, HCS_arr = Kou_HCS(fs, dT)
Kc = permeability(fs,pri_spac, mph)
print('hot crack susceptibility: ', HCS)


# plot solidfraction and HCS, permeability
fig2 = plt.figure()
ax2 = fig2.add_subplot(131)
ax2.plot(Tz_cp,fs)
plt.xlabel('temperature');plt.ylabel('solid fraction')
plt.legend(['cell','dendrite'])

ax3 = fig2.add_subplot(132)
ax3.plot(Tz_cp[1:-1],HCS_arr)
plt.xlabel('temperature');plt.ylabel('HCS')
plt.legend(['cell','dendrite'])

ax4 = fig2.add_subplot(133)
ax4.plot(Tz_cp,Kc)
plt.xlabel('temperature');plt.ylabel('permeability')
plt.legend(['cell','dendrite'])



#savemat('QoIs.mat',{'HCS':HCS})





