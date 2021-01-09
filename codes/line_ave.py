from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
from scipy.signal import savgol_filter
import matplotlib
import copy
import glob

num_reali = 12

num_traj = 1

num_qois = 17

#qoi_list = ['Ttip_arr','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','cqois']

qoi_list = ['Ttip_arr','ztip_qoi','pri_spac','sec_spac','interfl','HCS','Kc_ave','c1','c2','c3',\
            'c4','c5','c6','c7','c8','c9','c10']

keyword = ['angle75']    
    
list1 = glob.glob('*'+ keyword[0] + '*QoIs.mat')  


lists = [list1]


time1 = loadmat(list1[1])['time_qoi'].flatten()



time_list = [time1]



#ri = np.array([0.001225,0.001225,0.001225,0.001225])
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
    #degrees.append(str(beta)+' degree')
    d2c_line = qoi_arr[1,:];temp=copy.deepcopy(d2c_line); d2c_line_std = qoi_std[1,:]
    time_line = time1; dt = time_line[1] - time_line[0]
    vel_line = np.hstack((0,np.diff(d2c_line)/dt))
    dvel_line = np.hstack((0,np.diff(d2c_line_std)/dt))
    qoi_arr[1,:] = vel_line
    qoi_std[1,:] = dvel_line
    qoi_list = ['Ttip_arr','tip_vel','pri_spac','sec_spac','interfl','HCS','Kc_ave','c1','c2','c3',\
            'c4','c5','c6','c7','c8','c9','c10']
    savemat('line_' +'QoIs1.mat',{'time_qoi':time1,'dist':temp,'qoi_list':qoi_list,'qoi_arr':qoi_arr,'qoi_std':qoi_std})  




