#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:58:00 2020

@author: yigongqin
"""

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np
#from dsinput_gpu import phys_para, simu_para
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
from scipy.signal import savgol_filter
import matplotlib

import glob

ebwidth = 1.7

num_reali = 12

num_traj = 4

num_qois = 17

#qoi_list = ['Ttip_arr','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','cqois']

qoi_list = ['Ttip_arr','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','c1','c2','c3',\
            'c4','c5','c6','c7','c8','c9','c10']

#qoi_title = ['tip temperature','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','c1','c2','c3',\
           # 'c4','c5','c6','c7','c8','c9','c10']  
    
keyword = ['nss','nn','nl','sn']    
    
list1 = glob.glob('dirsolid_varGR_'+ keyword[0] + '*QoIs.mat')  
list2 = glob.glob('dirsolid_varGR_'+ keyword[1] + '*QoIs.mat')  
list3 = glob.glob('dirsolid_varGR_'+ keyword[2] + '*QoIs.mat')  
list4 = glob.glob('dirsolid_varGR_'+ keyword[3] + '*QoIs.mat')  

lists = [list1,list2,list3,list4]


time1 = loadmat(list1[1])['time_qoi'].flatten()
time2 = loadmat(list2[1])['time_qoi'].flatten()
time3 = loadmat(list3[1])['time_qoi'].flatten()
time4 = loadmat(list4[1])['time_qoi'].flatten()


time_list = [time1, time2, time3, time4]

time_true  = []
degrees = []
ri = np.array([0.001225,0.001225,0.001225,0.001225])
for k in range(num_traj):
    
    #macrodata = loadmat('macroGR_higherV3'+'.mat',squeeze_me=True)
   # rmax = macrodata['rmax']
    #beta = int(macrodata['beta'])
    num_time = len(time_list[k])
    
    qoi_real = np.zeros((num_reali,num_time))
    
    qoi_arr = np.zeros((num_qois,num_time))
    qoi_std = np.zeros((num_qois,num_time))

    for j in range(num_qois):
        
       qoi = qoi_list[j]
        
       for i in range(num_reali):
           
           if j<7:
               qoi_real[i,:] = loadmat(lists[k][i])[qoi].flatten()
           else:
               qoi_real[i,:] = loadmat(lists[k][i])['cqois'][j-7,:]
           if k==0 and j==2: plt.plot(qoi_real[i,100:])    
       qoi_arr[j,:] = np.mean(qoi_real, axis=0)
       qoi_std[j,:] = np.std(qoi_real, axis=0)/np.sqrt(num_reali)
    #qoi_std = np.zeros((num_qois,num_time))
    time_true.append((-ri[k]*1e6 + qoi_arr[1,:]-qoi_arr[1,0])*1e-3)
    #degrees.append(str(beta)+' degree')
    savemat(str(k)+ 'traj' +'QoIs.mat',{'time_qoi':time_list[k],'qoi_list':qoi_list,'qoi_arr':qoi_arr,'qoi_std':qoi_std})  
    
      #savemat('traj'+'k' +'.mat',{})  
    


#keyword = ['moderate Q and Vs','higher Q','higher Vs']



    
font = {'size'   : 12}

matplotlib.rc('font', **font)


saveflg = True

#dt = 0.001   #time_qoi[1]-time_qoi[0]

reducid = [70,70,70,70]

 #here time trye

macrodata = loadmat('macroGR_higherV_angle90.mat',squeeze_me=True)
Vp = macrodata['R_t']
t_macro = macrodata['t_macro']
#rmax = macrodata['rmax']

legends = ['dx = 0.4W0, W0 = 16.1d0','dx = 0.8W0, W0 = 16.1d0','dx = 1.2W0, W0 = 16.1d0','dx = 0.8W0, W0 = 11.3d0'] #degrees
colors = [(0.2, 0.4, 1, 1),(1, 0., 0., 1),(0.3, 1, 0.3, 1),(0.3, 0.3, 0.3, 1)]
transpen=0.2

ebcolors = [(0.2, 0.4, 1, transpen),(1, 0., 0., transpen),(0.3, 1,0.3, transpen),(0.3, 0.3, 0.3, transpen)]

def rsp( u ):
    
    u = u[:,:]
    ss, lenu = u.shape
    
    return np.reshape(u,(lenu))

def reduc(q_arr,q_std,time,reducid):
    
    
    return q_arr[:,reducid:], q_std[:,reducid:], time[reducid:]
    

def vel( z ,t ):
    
    v = np.diff(z)/np.diff(t)
    
    return np.hstack((0,v))

    

#rad_len = rsp(loadmat(dname[1])['ztip_qoi'])

fig2 = plt.figure(figsize=[12, 6])

ax2 = fig2.add_subplot(121)  # qoi #1
for i in range(num_traj):
    
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']
    ax2.errorbar(time_true[i],qoi_arr[0,:],yerr=qoi_std[0,:],c=colors[i]  ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('tip temperature (K)')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('tip temperature (K)')

ax2 = fig2.add_subplot(122)  # qoi #2
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']
    ax2.errorbar((time_list[i])[::5],vel(qoi_arr[1,::5],time_list[i][::5]),yerr=qoi_std[1,::5],c=colors[i]  ,elinewidth=ebwidth,ecolor = ebcolors[i])

ax2.plot((t_macro), (Vp))
plt.xlabel('time(s) ');plt.ylabel('tip velocity (um/s)')
plt.legend(['pulling velocity',legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('tip velocity (um/s)')

if saveflg == True: plt.savefig('tip_TV_c.png',dpi=400)
fig3 = plt.figure(figsize=[12, 6])


ax3 = fig3.add_subplot(121)  # qoi #3
for i in range(num_traj):

    
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])
    
    ax3.errorbar(time_redu,qoi_arr[2,:],yerr=qoi_std[2,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
#arr_lena = mpimg.imread('ps.png')
#imagebox = OffsetImage(arr_lena, zoom=0.1)
#ab = AnnotationBbox(imagebox, (-0.4, 5.1))
#ax3.add_artist(ab)
#arr_lena = mpimg.imread('ps2.png')
#i#magebox = OffsetImage(arr_lena, zoom=0.25)
#ab = AnnotationBbox(imagebox, (-0.6-0.12, 8.5))
#ax3.add_artist(ab)
#plt.draw()
plt.ylim(1,18)
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('primary spacing (um)')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('primary spacing (um)')

ax3 = fig3.add_subplot(122)  # qoi #4
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i]+70)
    
    ax3.errorbar(time_redu,qoi_arr[3,:],yerr=qoi_std[3,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
plt.ylim(0,5)
#arr_lena = mpimg.imread('ss.png')
#imagebox = OffsetImage(arr_lena, zoom=0.12)
#ab = AnnotationBbox(imagebox, (-0.4, 1.7))
#ax3.add_artist(ab)
#plt.draw()
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('secondary spacing (um)')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('secondary spacing (um)')
if saveflg == True: plt.savefig('spacing_c.png',dpi=400)



fig1 = plt.figure(figsize=[12,6])
ax1 = fig1.add_subplot(111)  # qoi #5
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])  
   
    #ax1.errorbar(time_redu,qoi_arr[4,:],yerr=qoi_std[4,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
    
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('interfacial length')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('interfacial length')

if saveflg == True: plt.savefig('interface_c.png',dpi=400)






fig6 = plt.figure(figsize=[12, 6])


ax6 = fig6.add_subplot(121)  # qoi #3
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax6.errorbar(time_redu,qoi_arr[5,:],yerr=qoi_std[5,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('HCS (K)')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('HCS (K)')

ax6 = fig6.add_subplot(122)  # qoi #4
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i]+50)    
    #ax6.set_yscale("log", nonposy='clip')
    #plt.ylim(1e-4,1e2)
    ax6.errorbar(time_redu,qoi_arr[6,:],yerr=qoi_std[6,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
plt.ylim(0,0.015)
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel('average permeability (um^2)')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('average permeability (um^2)')


if saveflg == True: plt.savefig('fs_c.png',dpi=400)    
#=========================================================================

######################=======================================qoi of c=============++##########################

fig4 = plt.figure(figsize=[20, 30])

ax4 =  plt.subplot2grid((3,2), (0,0))  # qoi #6
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])   
    ax4.errorbar(time_redu,qoi_arr[7,:],yerr=qoi_std[7,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
#ax4.plot(time_redu,3*np.ones(len(time_redu)))
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\bar{c}$')
plt.legend([legends[0],legends[1],legends[2],legends[3],'c_infty'],fontsize=9)
plt.title('average interfacial concentration (wt%)',fontsize=20)

ax4 =  plt.subplot2grid((3,2), (0,1))  # qoi #7
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[8,:],yerr=qoi_std[8,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\sigma^{2}_{c}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('variance of interfacial concentration  (wt%^2)',fontsize=20)



ax4 =  plt.subplot2grid((3,2), (1,0)) # qoi #3
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[9,:],yerr=qoi_std[9,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\bar{c}_{s,v}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('average solid concentration (wt%)',fontsize=20)

ax4 =  plt.subplot2grid((3,2), (1,1))  # qoi #4
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[10,:],yerr=qoi_std[10,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\sigma^{2}_{s,v}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('variance of solid concentration (wt%^2)',fontsize=20)




ax4 =  plt.subplot2grid((3,2), (2,0))  # qoi #6
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[11,:],yerr=qoi_std[11,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
#ax4.plot(time_redu,3*np.ones(len(time_redu)))
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\bar{c}_{l,v}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)#,'c_infty'])
plt.title('average solid concentration at the interface (wt%)',fontsize=20)

ax4 =  plt.subplot2grid((3,2), (2,1))  # qoi #7
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[12,:],yerr=qoi_std[12,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])
plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\sigma^{2}_{l,v}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('variance solid concentration at the interface (wt%^2)',fontsize=20)


if saveflg == True: plt.savefig('concentration_c.png',dpi=400)




fig4 = plt.figure(figsize=[20, 20])





ax4 =  plt.subplot2grid((2,2), (0,0))  # qoi #3
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[13,:],yerr=qoi_std[13,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\bar{c}_{s,i}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('average liquid concentration (wt%)',fontsize=20)

ax4 =  plt.subplot2grid((2,2), (0,1)) # qoi #4
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[14,:],yerr=qoi_std[14,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\sigma^{2}_{s,i}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('variance of liquid concentration (wt%^2)',fontsize=20)

#if saveflg == True: plt.savefig('concentration',dpi=400)





ax4 =  plt.subplot2grid((2,2), (1,0)) # qoi #3
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[15,:],yerr=qoi_std[15,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\bar{c}_{l,i}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('average liquid concentration at the interface (wt%)')

ax4 =  plt.subplot2grid((2,2), (1,1))  # qoi #4
for i in range(num_traj):
    qoi_arr = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_arr']
    qoi_std = loadmat(str(i)+ 'traj' +'QoIs.mat')['qoi_std']  
    qoi_arr, qoi_std, time_redu = reduc(qoi_arr, qoi_std, time_true[i], reducid[i])    
    ax4.errorbar(time_redu,qoi_arr[16,:],yerr=qoi_std[16,:],c=colors[i] ,elinewidth=ebwidth,ecolor = ebcolors[i])

plt.xlabel('distance from the center (mm)', fontsize = 15);plt.ylabel(r'$\sigma^{2}_{l,i}$')
plt.legend([legends[0],legends[1],legends[2],legends[3]],fontsize=9)
plt.title('variance liquid concentration at the interface (wt%^2)')

if saveflg == True: plt.savefig('liquid_concentration_c.png',dpi=400)



######################=======================================qoi of c=============++##########################



    


    
    
    
    
    
    

