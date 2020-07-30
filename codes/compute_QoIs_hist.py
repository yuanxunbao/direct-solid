#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:33:02 2020

@author: yigongqin
"""


import numpy as np
from dsinput_gpu import phys_para, simu_para
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
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

Tt_arr = np.linspace(0,Ttd,nts+1)

Ti = 931.20
Te = 821

Ntip = 1

mph = 'dendrite'
#mph = 'cell'

## ======================= load data ==========================##
# loadfile1 = 'move_eta004.mat'
loadfile1 = 'dirsolid_gpu_noise4.00E-02_misori0_lx2812.5_nx2000_asp0.5_seed192'


op = (loadmat(loadfile1)['order_param'])
cc = (c_infty*loadmat(loadfile1)['conc'])
#zz_mv = (loadmat(loadfile1)['zz_mv'])

# set evaluation times
sample = np.arange(5,nts+1,1)
times = Tt_arr[sample]
num_sam = len(sample)
pri_arr = np.zeros(num_sam);sec_arr = np.zeros(num_sam);
inter_arr = np.zeros(num_sam);var_arr = np.zeros(num_sam);
hcs_arr = np.zeros(num_sam);perm_arr = np.zeros(num_sam);
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

for num_time in range(len(times)):
    
    time_ = times[num_time]
    print('time now is: ', time_,'s')
    sam = sample[num_time]
    phi = op[:,sam].reshape((nx,nz),order='F').T
    conc = cc[:,sam].reshape((nx,nz),order='F').T
    #zz = zz_mv[:,-1].reshape((nx,nz),order='F').T
    zz = (loadmat(loadfile1)['zz']).T

    Tz =  Ti + G*( zz[:,3] - R*time_ )
    
    print('the range of z coordinates: ', zz[0,1],'um ~',zz[-1,3],'um')
    T_bottom = Tz[0]; T_top = Tz[-1]
    print('the range of temperature: ', T_bottom,'K ~',T_top,'K')
    #print('\n')
    
    
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
    Ttip = Ti + G*( ztip - R*time_ )
    print('tip temperature: ',Ttip ,'K')
    
    # dendrite structure
    inter_len = interf_len(phi)*dxd**2*1e-6
    print('length of interface: ', inter_len)
    #print('\n')
    
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
    fs = smooth_fs(fs,len(fs) if len(fs)%2==1 else len(fs)-1)
    HCS, HCS_arr = Kou_HCS(fs, dT)
    Kc = permeability(fs,pri_spac, mph)
    print('hot crack susceptibility: ', HCS)
    Kc_bar = np.mean(Kc)
    print('average permeability', Kc_bar )
    print('\n')
    pri_arr[num_time] = pri_spac;sec_arr[num_time] = sec_spac
    inter_arr[num_time] = inter_len;var_arr[num_time] = var_phi
    hcs_arr[num_time] = HCS;perm_arr[num_time] = Kc_bar
    '''
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
    ax4.plot(Tz_cp,np.log10(Kc))
    plt.xlabel('temperature');plt.ylabel('permeability log(Kc)')
    plt.legend(['cell','dendrite'])
    '''

# plot time history
    # plot solidfraction and HCS, permeability
fig2 = plt.figure()
ax2 = fig2.add_subplot(231)
ax2.plot(times,pri_arr)
plt.xlabel('time');plt.ylabel('primary spacing (um)')

ax3 = fig2.add_subplot(232)
ax3.plot(times,sec_arr)
plt.xlabel('time');plt.ylabel('secondary spacing (um)')

ax4 = fig2.add_subplot(233)
ax4.plot(times,inter_arr)
plt.xlabel('time');plt.ylabel('interfacial length')

ax5 = fig2.add_subplot(234)
ax5.plot(times,var_arr)
plt.xlabel('time');plt.ylabel('variation in phi')

ax6 = fig2.add_subplot(235)
ax6.plot(times,hcs_arr)
plt.xlabel('time');plt.ylabel('HCS')

ax7 = fig2.add_subplot(236)
ax7.plot(times,perm_arr)
plt.xlabel('time');plt.ylabel('permeability log(Kc)')


#savemat('QoIs.mat',{'HCS':HCS})





