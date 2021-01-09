#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:08:00 2020

@author: yigongqin
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from math import pi

#num_gpu = 4
#npx  = 2
#npy = npx

#filebase = sys.argv[1];
#nx = int(sys.argv[2])
ratio = 0.5 #float(sys.argv[3])
#ny = int(nx*ratio)
#print(nx,ny)
lx = 40
ly = lx*ratio
print('the limits for geometry lx, ly: ',lx,ly)

## plot the QoIs,
# typical range: (0) alphaB [-0.85,-0.65], (1) tip_vel [0, 8000], (2) pri_spac [5,15], (3) sec_spac [0,7], 
# (4) interf_len [], (5) HCS [], (6) Kc_ave[], (7) solid_conc []

names=['Ttip','tip_vel', 'pri_spac','sec_spac','interf_len','HCS', 'Kc_ave','cqois','cqois','cqois']
range_l = [885, 0, 0, 0, 0, 20, 0, 0, 0, 0]
range_h = [920, 170000, 2, 1, 15, 100, 10e-4, 3, 3, 3]
titles=['$tip\ temperature (K)$','$tip\ velocity (\mu m/s)$', '$primary\ spacing\ (\mu m)$','$secondary\ spacing\ ({\mu}m)$',\
'$interfacial\ length$','$HCS\ (K)$', '$permeability\ ({\mu m}^{2})$','','','$solid\ concentration\ (wt\%)$']

qid = 1
qoi_name = names[qid]
vmin = np.float64(range_l[qid])
vmax = np.float64(range_h[qid])
# load line model qoi data 


colors = [(0.2, 0.4, 1, 1),(1, 0., 0., 1),(0.3, 1, 0.3, 1),(0.3, 0.3, 0.3, 1)]
transpen=0.25;ebwidth = 3; m_size=30; tc=1.96
ebcolors = [(0.2, 0.4, 1, transpen),(1, 0., 0., transpen),(0.3, 1,0.3, transpen),(0.3, 0.3, 0.3, transpen)]

## need DNS/line/macrodata
linedata0 = sio.loadmat('line_QoIs0.mat',squeeze_me = True)
linedata1 = sio.loadmat('line_QoIs1.mat',squeeze_me = True)
linedata2 = sio.loadmat('line_QoIs2.mat',squeeze_me = True)
data = sio.loadmat('dns_QoIs0.mat',squeeze_me = True)
macrodata=sio.loadmat('AM2.mat',squeeze_me = True)
X_arr=macrodata['X_arr']
Y_arr=macrodata['Y_arr']
line_angle=macrodata['line_angle']
line_xst=macrodata['line_xst']
line_yst=macrodata['line_yst']
print(line_xst);print(line_yst)
for qid in range(8):
  if qid ==7: qid +=2
  qoi_name = names[qid]
  vmin = np.float64(range_l[qid])
  vmax = np.float64(range_h[qid])
  fig1 = plt.figure(figsize=[6, 12]) 
  fig1.suptitle(titles[qid], fontsize=16)
  xB = data['xB']; zB = data['zB']; qoi = data['qoi_arr'][qid,:];qoi_std =data['qoi_std'][qid,:]; 
  line_id = 0; line_dist=linedata0['dist']; line_qoi=linedata0['qoi_arr'][qid,:];line_std=tc*linedata0['qoi_std'][qid,:];
  bool_arr = np.absolute( np.arctan((zB-line_yst[line_id])/(xB-line_xst[line_id]))-line_angle[line_id]) <0.01
  xB = xB[bool_arr]; zB = zB[bool_arr]; qoi=qoi[bool_arr];qoi_std=qoi_std[bool_arr]; 
  dist = np.sqrt((zB-line_yst[line_id])**2+(xB-line_xst[line_id])**2); print('distance', dist)
  ax0 = fig1.add_subplot(311)
  if qid==1 or qid==0: ax0.errorbar(line_dist,line_qoi,yerr=line_std,c=colors[0],elinewidth=ebwidth,ecolor = ebcolors[0])
  else: ax0.errorbar(line_dist-1.5,line_qoi,yerr=line_std,c=colors[0],elinewidth=ebwidth,ecolor = ebcolors[0])
  if qid==0: ax0.scatter(dist+1.5,qoi,s=m_size,c='r',marker='x');
  else:ax0.scatter(dist,qoi,s=m_size,c='r',marker='x');
  plt.xlim(0,ly);plt.ylim(range_l[qid],range_h[qid])
  plt.xlabel('traveled distance');plt.legend(['DNS simulation (static)','$line\ model\ 86^{\circ}$'])

  xB = data['xB']; zB = data['zB']; qoi = data['qoi_arr'][qid,:];qoi_std =data['qoi_std'][qid,:];
  line_id = 1; line_dist=linedata1['dist']; line_qoi=linedata1['qoi_arr'][qid,:];line_std=tc*linedata1['qoi_std'][qid,:];
  bool_arr = np.absolute( np.arctan((zB-line_yst[line_id])/(xB-line_xst[line_id]))-line_angle[line_id]) <0.01
  xB = xB[bool_arr]; zB = zB[bool_arr]; qoi=qoi[bool_arr];qoi_std=qoi_std[bool_arr];
  dist = np.sqrt((zB-line_yst[line_id])**2+(xB-line_xst[line_id])**2); print('distance', dist)
  ax1 = fig1.add_subplot(312)
  if qid==1 or qid==0: ax1.errorbar(line_dist,line_qoi,yerr=line_std,c=colors[3],elinewidth=ebwidth,ecolor = ebcolors[3])
  else: ax1.errorbar(line_dist-1.5,line_qoi,yerr=line_std,c=colors[3],elinewidth=ebwidth,ecolor = ebcolors[3])
  if qid==0: ax1.scatter(dist+1.5,qoi,s=m_size,c='r',marker='x'); 
  else:ax1.scatter(dist,qoi,s=m_size,c='r',marker='x'); 
  plt.xlim(0,ly);plt.ylim(range_l[qid],range_h[qid])
  plt.xlabel('traveled distance');plt.legend(['DNS simulation (static)','$line\ model\ 75^{\circ}$'])

  xB = data['xB']; zB = data['zB']; qoi = data['qoi_arr'][qid,:];qoi_std =data['qoi_std'][qid,:];
  line_id = 2; line_dist=linedata2['dist']; line_qoi=linedata2['qoi_arr'][qid,:];line_std=tc*linedata2['qoi_std'][qid,:];
  bool_arr = np.absolute( np.arctan((zB-line_yst[line_id])/(xB-line_xst[line_id]))-line_angle[line_id]) <0.01
  xB = xB[bool_arr]; zB = zB[bool_arr]; qoi=qoi[bool_arr];qoi_std=qoi_std[bool_arr];
  dist = np.sqrt((zB-line_yst[line_id])**2+(xB-line_xst[line_id])**2); print('distance', dist)
  ax2 = fig1.add_subplot(313)
  if qid==1 or qid==0: ax2.errorbar(line_dist,line_qoi,yerr=line_std,c=colors[2],elinewidth=ebwidth,ecolor = ebcolors[2])
  else: ax2.errorbar(line_dist-1.5,line_qoi,yerr=line_std,c=colors[2],elinewidth=ebwidth,ecolor = ebcolors[2])
  if qid==0: ax2.scatter(dist+1.5,qoi,s=m_size,c='r',marker='x'); 
  else:ax2.scatter(dist,qoi,s=m_size,c='r',marker='x');
  plt.xlim(0,ly);plt.ylim(range_l[qid],range_h[qid])
  plt.xlabel('traveled distance');plt.legend(['DNS simulation (static)','$line\ model\ 53^{\circ}$'])

  plt.savefig('HG_qoi_' + qoi_name +'.png',dpi=400)
  plt.close()

'''

fig0 = plt.figure(constrained_layout = True)
spec0 = fig0.add_gridspec(ncols=npx, nrows=npy)
for row in range(npy):
    for column in range(npx):

         i = (npy-1-row)*npx + column 
         filename = filebase +str(i)+'hawd100.mat'
         data = sio.loadmat(filename,squeeze_me = True)
         num_box =data['num_box']
         if num_box !=0:
           u =data['c_win']

           ax = fig0.add_subplot(spec0[row,column])
           extents =(0, 40, 0, 40)
           cs=plt.imshow(u,cmap=plt.get_cmap('jet'),origin='lower',extent= extents)
           #cbar = fig0.colorbar(cs)
           plt.clim(0, 5)
cbar = fig.colorbar(cs) 
plt.savefig('windows_conc' + '.png',dpi=600)
plt.close()

'''




