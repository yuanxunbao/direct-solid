#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:35:17 2020

@author: yigongqin
"""

import numpy as np
#from dsinput_gpu import phys_para, simu_para
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
from scipy.signal import savgol_filter

dname1 = 'dirsolid_varGR_noise0.04821_misori0_lx36.20_nx199_asp10_ictype1_U0-1.00_QoIs.mat'
dname2 = 'dirsolid_varGR_noise0.04121_misori0_lx36.20_nx199_asp10_ictype1_U0-1.00_QoIs.mat'
dname3 = 'dirsolid_varGR_noise0.0423_misori0_lx36.20_nx199_asp10_ictype1_U0-1.00_QoIs.mat'
dname4 = 'dirsolid_varGR_noise0.04502_misori0_lx36.20_nx199_asp10_ictype1_U0-1.00_QoIs.mat'
sets =4
dname = np.array([dname1, dname2, dname3, dname4])

def rsp( u ):
    
    u = u[:,:-1]
    ss, lenu = u.shape
    
    return np.reshape(u,(lenu))


def vel( z ,t ):
    
    v = np.diff(z)/np.diff(t)
    
    return np.hstack((0,v))

'''

    
time  = rsp(loadmat(dname1)['time_qoi'])


fig2 = plt.figure(figsize=[12, 6])

ax2 = fig2.add_subplot(121)  # qoi #1
for i in range(sets):
    
    ax2.plot(time,rsp(loadmat(dname[i])['Ttip_arr']))

plt.xlabel('time (s)');plt.ylabel('tip temperature (K)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('tip temperature (K)')

ax2 = fig2.add_subplot(122)  # qoi #2
for i in range(sets):
    ax2.plot(time,vel(rsp(loadmat(dname[i])['ztip_qoi']),time))

plt.xlabel('time (s) ');plt.ylabel('tip velocity (um/s)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('tip velocity (um/s)')

#plt.savefig('tip',dpi=400)

fig3 = plt.figure(figsize=[12, 6])


ax3 = fig3.add_subplot(121)  # qoi #3
for i in range(sets):
    
    ax3.plot(time,rsp(loadmat(dname[i])['pri_spac']))

plt.xlabel('time (s)');plt.ylabel('primary spacing (um)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('primary spacing (um)')

ax3 = fig3.add_subplot(122)  # qoi #4
for i in range(sets):
    
    ax3.plot(time,rsp(loadmat(dname[i])['sec_spac']))

plt.xlabel('time (s)');plt.ylabel('secondary spacing (um)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('secondary spacing (um)')
#plt.savefig('spacing',dpi=400)

fig4 = plt.figure(figsize=[12, 6])
ax4 = fig4.add_subplot(131)  # qoi #5
for i in range(sets):
    
    ax4.plot(time,rsp(loadmat(dname[i])['interfl']))

plt.xlabel('time (s)');plt.ylabel('total interfacial length')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('total interfacial length')
ax4 = fig4.add_subplot(132)  # qoi #6
for i in range(sets):
    
    ax4.plot(time,rsp((loadmat(dname[i])['cqois'])[[0],:]))
ax4.plot(time,3*np.ones(len(time)))
plt.xlabel('time (s)');plt.ylabel('average interfacial concentration (wt%)')
plt.legend(['realization1','realization2','realization3','realization4','c_infty'])
plt.title('average interfacial concentration (wt%)')

ax4 = fig4.add_subplot(133)  # qoi #7
for i in range(sets):
    
    ax4.plot(time,rsp((loadmat(dname[i])['cqois'])[[2],:]))
plt.xlabel('time (s)');plt.ylabel('average solid concentration (wt%)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('average solid concentration (wt%)')
#plt.savefig('interface',dpi=400)


fig5 = plt.figure(figsize=[12, 6])


ax5 = fig5.add_subplot(121)  # qoi #3
for i in range(sets):
    
    ax5.plot(time,rsp((loadmat(dname[i])['cqois'])[[6],:]))

plt.xlabel('time (s)');plt.ylabel('interfacial concentration variation (wt%)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('interfacial concentration variation (wt%)')

ax5 = fig5.add_subplot(122)  # qoi #4
for i in range(sets):
    
    ax5.plot(time,rsp((loadmat(dname[i])['cqois'])[[5],:]))

plt.xlabel('time (s)');plt.ylabel('solid concentration variation (wt%)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('solid concentration variation (wt%)')
#plt.savefig('variation',dpi=400)
fig6 = plt.figure(figsize=[12, 6])


ax6 = fig6.add_subplot(121)  # qoi #3
for i in range(sets):
    
    ax6.plot(time,rsp((loadmat(dname[i])['HCS'])))

plt.xlabel('time (s)');plt.ylabel('HCS (K)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('HCS (K)')

ax6 = fig6.add_subplot(122)  # qoi #4
for i in range(sets):
    
    ax6.plot(time,rsp((loadmat(dname[i])['Kc_ave'])))

plt.xlabel('time (s)');plt.ylabel('average permeability (um^2)')
plt.legend(['realization1','realization2','realization3','realization4'])
plt.title('average permeability (um^2)')


#plt.savefig('fs',dpi=400)

'''

fs  = (loadmat(dname1)['fs_arr'][:,189])
fs = savgol_filter(fs, 245, 3)
plt.plot(fs)


# load data set