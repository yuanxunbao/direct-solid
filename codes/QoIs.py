#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:17:57 2020

@author: yigongqin
"""


import numpy as np
from scipy.signal import savgol_filter
from numpy import int32 
from math import pi
from numpy.linalg import norm

# array computation: axis=0 vetical; axis=1, horizontal. After doing one np function, the result is 1D array
# all index N should follow python index rules (start from 0)
##==================== tip track and domain crop =========================##


def ztip_Ntip( phi, zz, Ntip):   # track tip location from solid part
    
    phi_tip = phi[Ntip,:]
    
    
    while np.mean(phi_tip)>-0.99:
        
        Ntip += 1
        
        phi_tip = phi[Ntip,:]
    
    
    return  zz[Ntip,3], Ntip


# for now, just assume we already have information of every time step. So input a location 1D array(z and N) and a sampling time array

def steady_state_region( ztip_arr, Ntip_arr, t_arr):
    
    dts = t_arr[1] - t_arr[0]
    vtip = np.diff(ztip_arr)/dts            # https://numpy.org/doc/stable/reference/generated/numpy.diff.html 
    accele = np.diff( vtip )/vtip[-1]
    steady = np.argwhere(accele<0.0001)        # https://numpy.org/doc/1.18/reference/generated/numpy.argwhere.html
    tindex = steady[0] +1     
    
    vtip_arr = np.hstack((vtip,vtip[-1]))
    
    return  t_arr[tindex], Ntip_arr[tindex], vtip_arr  # vtip lacks one 1 element



def crop_domain( Ntip, cgrid, Nss, z, phi):    #Ntip is larger than Nss
    # for moving frame, the domain is automatically cropped.
    # for now, domain is cut manually, crop is necessary for the QoIs following
    
    # crop extra liquid portion (ztip,ntip)
    Nup = Ntip + cgrid
    # crop unsteady part ( Nss )
  
    return  z[Nss:Nup], phi[Nss:Nup,:]


def tcp( phi, Ntip, cgrid ):
    
    Nup = Ntip + cgrid
    return phi[:Nup,:]

    
def tcpa( phi, Ntip, cgrid ):
    
    Nup = Ntip + cgrid
    return phi[:Nup]

def gener_win_spac():
    
    
    return

##==================== Eutectic region properties =========================##
# here phi/c should be cropped by steady state and Te lines
# Te is eutectic temperature 

def eutectic_Vfrac(phi, Nss, T, Te):
    
    nz, nx = phi.shape
    
    Neu = np.argmin( np.absolute(T-Te) )
    
    feu = (1- phi[Nss:Neu,:] )/2
    
    return np.mean(feu)

def solute_variability(c, Nss, T, Te):
    
    nz, nx = c.shape
    
    Neu = np.argmin( np.absolute(T-Te) )
    
    ceu = c[Nss:Neu,:]
    
    return np.std(ceu)


##==================== cellular/dendritic structure =========================##
# the only input for this section is phi that has been vertically cropped      

def interf_len(phi):
    
    mp, np = phi.shape
    
    inter_len = np.sum( (1-phi**2) )
    Lf = inter_len/(np*np)
    
    return Lf # one number



def conc_var(phi,c):
    
    s_mask = (phi>0)*1
    l_mask = 1 - s_mask
    ns = np.sum(s_mask)
    nl = np.sum(l_mask)
    
    Lf = np.sum( 1-phi**2 )    
    
    cb_ave = np.sum( (1-phi**2)*c )/Lf
    cb_var = np.sqrt( np.sum( (1-phi**2)*(c-cb_ave)**2 )/Lf )
    
    cs = np.sum( c*s_mask )/ns
    cs_var = np.sqrt(np.sum( (c-cs)**2*s_mask )/ns)
    cs_var2 = np.sqrt(np.sum( (c-cs)**2*s_mask*(1-phi**2) )/np.sum(s_mask*(1-phi**2)) )
    
    cl = np.sum( c*l_mask )/nl
    cl_var = np.sqrt(np.sum( (c-cl)**2*l_mask )/nl)
    cl_var2 = np.sqrt(np.sum( (c-cl)**2*l_mask*(1-phi**2) )/np.sum(l_mask*(1-phi**2)) )
    
    
    return np.array[cb_ave, cb_var, cs, cs_var, cs_var2, cl, cl_var, cl_var2]



def phi_xstat(phi,Ntip):
    
    phi_cp = tcp(phi,Ntip,10)
    mui = np.mean(phi_cp, axis=0)            #(nx,)
    sigmai = np.std(phi_cp, axis=0)          #(nx,)
    var_phi = np.mean(sigmai)
    
    return   mui, sigmai, var_phi  # two 1D arrays and 1 number


def spacings(phi, Ntip, lxd, dxd, mph):
    
    
   # mui, sigmai, var_phi = phi_xstat(phi,Ntip)
    if Ntip>30:
        mui = np.mean(phi[Ntip-30:Ntip+10,:], axis=0)
    else: mui = np.mean(phi[:Ntip+10,:], axis=0)
    cells = identify_peak(mui)
    pri_spac = primary_spacing(mui, lxd, dxd)
    #print(lxd,pri_spac)
    if mph == 'cell':
        return pri_spac, 0.0
    
    else:
        
        # for secondary need to cut
        if Ntip>100:        
            phi_cp = tcp(phi,Ntip,-100)   # originally 150
        else:
            phi_cp = tcp(phi,Ntip,0)
            
        sigmai = np.std(phi_cp, axis=0)          #(nx,)
        # further, need to do some filtering for sigmai to ensure sidebranching is big enough
        sec_spac_arr, sides = secondary_spacing(cells, sigmai, phi_cp, dxd)
        #print(sec_spac_arr)
        sec_spac_arr = np.sort(sec_spac_arr)
        ns = int( len(sec_spac_arr)/4 )
        sec_spac = np.mean( sec_spac_arr[ns:len(sec_spac_arr)-ns] )
        
        return pri_spac, sec_spac


def primary_spacing(mui,lxd,dxd):    

    nprim,lenpeak = crosspeak_counter(mui, dxd, -0.3)
    
    return  lxd/nprim  # a number


def secondary_spacing(cells, sigmai, phi_cp, dxd):
    
    ncells =len(cells)
    nsides = 2*len(cells)
    sides = np.zeros(nsides,dtype=int)
    seco_spac = np.zeros(nsides)
    
    mid = ((cells[1:]+cells[:-1])/2).astype(int)
    coors = np.zeros(nsides-1,dtype=int)
    coors[0] = cells[0]
    
    for jj in range(1, ncells):
        
        coors[2*jj-1] = mid[jj-1]
        coors[2*jj] = cells[jj]
       
        
    coor = np.hstack((0,coors,len(sigmai)))  
    #print(cells);print(sigmai);print(coor)
    for i in range(nsides):
             
        sides[i] = np.argmax(sigmai[coor[i]:coor[i+1]]) + coor[i]   #index where we have sidebraching ( the largest variation in phi along z axis)
        
    for ii in range(nsides):
        
        phi_j = phi_cp[:,sides[ii]] 
        nseco,lside = crosspeak_counter(phi_j, dxd, 0)
        seco_spac[ii] = lside/nseco
        
    return seco_spac, sides    



def crosspeak_counter(u, dx, indicator): # u is a 1D array
    

    intersec = np.argwhere(np.diff(np.sign( u - indicator)))
    dist = intersec[-1]-intersec[0]
    
    return int( len(intersec) /2), dist*dx 

def identify_peak(u): # u is a 1D array (nx,)
    
    peak = (u[1:-1]>u[2:]) & (u[1:-1]>u[:-2])*1
    
    index = np.argwhere(peak) +1
    
    return np.reshape(index,(len(index)))
            

                      
#===================== solid fraction fs ==============================#
# from here, the phi and T should be cropped from whole domain or get from the moving frame
# T can always be got from cropped z    
def solid_frac(phi, Ntip, Te, Tz):
    # T here should be a (nz,) array, same as z variable
    
    mask = 1*(Tz>Te)
        
    fs = np.mean( (phi +1)/2, axis=1)
    
    return fs*mask + (1-mask)    # return fs should be a (nz,) array, same as Tz, fs and Tz are one pair

def smooth_fs(fs,winds):
    
    fs = savgol_filter(fs, winds, 3)  #winds is smaller than size of fs
    
    return fs



def Kou_HCS(fs, dT):   # input interval of temperature level sets
    
    sq_fs = np.sqrt(fs)
    dTdfsh = (2*dT )/(sq_fs[2:]-sq_fs[:-2])
    
    HCS_K_arr = np.absolute( dTdfsh )
    HCS_K = np.amax( HCS_K_arr )
    
    return HCS_K, HCS_K_arr  #a number


def permeability(fs,lamd, mph):#lambda here is primary spacing
    
    fs2 = fs**2
    if mph =='cell':
        
        return lamd**2*( 1- fs2 )**3/( 180*fs2 )
    
    else:
        
        mask_l = 1*(fs<0.25)
        mask_h = 1*(fs>0.35)
        
        Kd_l = 0.074*( np.log( fs**-1 ) -1.49 +2*fs -0.5*fs**2 ) *lamd**2
        Kd_m = 2.05e-7*( (1-fs)/fs )**10.739 *lamd**2
        Kd_h = 3.75e-4*(1-fs)**2 *lamd**2
        
        return mask_l*Kd_l + mask_h*Kd_h + (1 - mask_l - mask_h)*Kd_m        # 1D array
    




##  ================= misorientation ================  ##
    

def tilted_angle(ph_up, ph_down, Nd, Npri, dire): # Npri is the number of grids of primary spacing
    
    
    shift = 0
    sum_shift = 0
    Nzz, Nxx = ph_up.shape
    diff = norm(ph_up-ph_down)
    
    if dire == 1:
        for jj in range(Nzz):
            
            for  ii in range(Npri):
                
                ph_up[jj,:] = np.hstack((ph_up[jj,-1],ph_up[jj,:-1]))
                diff_temp = norm(ph_up[jj,:]-ph_down[jj,:])
                
                if diff_temp<diff:
                    diff = diff_temp
                    shift = ii
            
            sum_shift += shift                  
        alpha = np.arctan(sum_shift/Nzz/Nd)
        #print(sum_shift/Nzz)    
        
    elif dire == -1:
        for jj in range(Nzz):
            for  ii in range(Npri):
                
                ph_up[jj,:] = np.hstack((ph_up[jj,1:],ph_up[jj,0]))
                diff_temp = norm(ph_up[jj,:]-ph_down[jj,:])
                
                if diff_temp<diff:
                    diff = diff_temp
                    shift = ii
            sum_shift += shift                  
        alpha = np.arctan(sum_shift/Nzz/Nd)                
        #alpha = np.arctan( (Npri-shift)/Nd )
    else: print('invalid orientation occured!')
    
    return alpha*180/pi


def plumb(phi, alpha):   # phi here should be already post-processed.

    alpha = alpha*pi/180
    
    tana = np.tan(alpha)
    Nz, Nx = phi.shape
    phi_pl = 0.0*phi
    for ii in range(Nz):
    # find neighboors for grids
        j = np.arange(Nx)
        jn = j - ii*tana
        jnm = np.floor(jn)
        jnp = np.ceil(jn)
        am = jn - jnm
    # interpolate phi from the above coordinates
        jnm = int32(jnm%Nx); jnp = int32(jnp%Nx)
        phi_pl[ii,:] = (1-am)*(phi[ii,jnm]) + am*(phi[ii,jnp])
        

    
    return phi_pl   # should be the same as veltical cells/dendrites
    



































