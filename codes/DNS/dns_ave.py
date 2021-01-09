import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
import matplotlib
import sys
import glob

num_reali = 1

num_gpu = 4

num_qois = 17

#qoi_list = ['Ttip_arr','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','cqois']
#['alphaB','tip_vel', 'pri_spac','sec_spac','interf_len','HCS', 'Kc_ave','cqois']
qoi_list = ['Ttip','tip_vel','pri_spac','sec_spac','interf_len','HCS','Kc_ave','c1','c2','c3',\
            'c4','c5','c6','c7','c8','c9','c10']
filebase = sys.argv[1]
keyword = ['seed128']#, 'seed2', 'seed3']

list1 = glob.glob(filebase + keyword[0] + '*hawd100.mat')   
#list2 = glob.glob(filebase + keyword[1] + '*hawd100.mat')
#list3 = glob.glob(filebase + keyword[2] + '*hawd100.mat')

lists = [list1]#,list2,list3]

xB_list= []; zB_list=[];
for kk in range(num_gpu):
     xB_list.extend(loadmat(lists[0][kk])['xB'].flatten())
     zB_list.extend(loadmat(lists[0][kk])['zB'].flatten())

num_box =  len(xB_list)
print('total number of boxes',num_box)
#print('the x coors', xB_list)
#print('the z coors', zB_list)
    
qoi_real = np.zeros((num_reali,num_box))
    
qoi_arr = np.zeros((num_qois,num_box))
qoi_std = np.zeros((num_qois,num_box))

for j in range(10):#num_qois):
        
     qoi = qoi_list[j]; print('the current qoi: ', qoi)
        
     for i in range(num_reali):
        print('load realization: ', keyword[i])

        qoi_gpu = []
        for kk in range(4):

           dd = loadmat(lists[i][kk],squeeze_me=True)
           if dd['num_box']!=0:
             if j<7:
               qoi_gpu.extend( dd[qoi] )
             else:
               qoi_gpu.extend( dd['cqois'][j-7,:])
        qoi_real[i,:] = np.array(qoi_gpu)
     qoi_real[qoi_real ==0.0] = np.nan
    # means = np.nanmean(data[:, 1:], axis=1)
     qoi_arr[j,:] = np.nanmean(qoi_real, axis=0); print('the average', qoi_arr[j,:])
     qoi_std[j,:] = np.nanstd(qoi_real, axis=0)/np.sqrt(num_reali)

# use HCS as standard, filter out points
xB = np.array(xB_list); zB = np.array(zB_list);
#filt = np.isnan(qoi_arr[5,:])
#xB=xB[filt]; zB=zB[filt]; qoi_arr= qoi_arr[:,filt]; qoi_std= qoi_std[:,filt]
print(xB.shape);print(qoi_arr.shape)
savemat('dns_QoIs0.mat',{'xB':xB,'zB':zB, 'qoi_list':qoi_list,'qoi_arr':qoi_arr,'qoi_std':qoi_std})  





