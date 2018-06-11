import scipy.linalg as linalg
import numpy as np
import math
from scipy import integrate
import scipy.linalg.interpolative as sli

import time
import sys
import os
import re
# 
def print_sim_env(chi,d,N,i_delta,chi_max,J,alp,Dz,r,hz,hst,g) :
    wline=''
    wline += 'chi    ='+str(chi)+"\n"
    wline += 'd      ='+str(d)+"\n"
    wline += 'N      ='+str(N)+"\n"
    wline += 'i_dt   ='+str(np.imag(i_delta))+"\n"
    wline += 'chi_max='+str(chi_max)+"\n"
    wline += 'J      ='+str(J)+"\n"
    wline += 'alpha  ='+str(alp)+"\n"
    wline += 'Dz     ='+str(Dz)+"\n"
    wline += 'Delta  ='+str(r)+"\n"
    wline += 'hz     ='+str(hz)+"\n"
    wline += 'hst    ='+str(hst)+"\n"
    wline += 'g      ='+str(g)+"\n"
    f = open('parameter.txt','w')
    f.write(wline)
    f.close()


def dump_write(R,K,S,L,T,M,TE,TO,Tl,Tlw,Tr,Trw) :
    np.savez('dump.npz',R=R,K=K,S=S,L=L,T=T,M=M,TE=TE,TO=TO,Tl=Tl,Tlw=Tlw,Tr=Tr,Trw=Trw)

def pack_parameters(chi,d,chi_max,N,i_delta,t_max,J,alp,Dz,r,g,hz,hst) :
    P = np.savez('params.npz',chi=chi,d=d,chi_max=chi_max,N=N,i_delta=i_delta,t_start=t_max,J=J,alp=alp,Dz=Dz,r=r,g=g,hz=hz,hst=hst)

def read_dump() :
    try : 
      temp = np.load('dump.npz') 
    except:
      print('Error : Failure of reading dumping files')
      sys.exit()
    return temp['R'],temp['K'],temp['S'],temp['L'],temp['T'],temp['M'],temp['TE'],temp['TO'],temp['Tl'],temp['Tlw'],temp['Tr'],temp['Trw']

def unpack_parameters(read_dump_file) :
    if(read_dump_file) :
      try :
        temp = np.load('params.npz')
        chi = temp['chi']; d = temp['d']; chi_max = temp['chi_max']; N = temp['N']; i_delta = temp['i_delta']
        t_start = temp['t_start']; J = temp['J']; alp = temp['alp']; r = temp['r']; g = temp['g'];
        hz = temp['hz']; hst = temp['hst']; Dz = temp['Dz']; 
      except :
        print('Error : Failure of reading a parameter file.')
        sys.exit()
    return chi,d,chi_max,N,i_delta,t_start,J,alp,Dz,r,hz,hst,g
######
######
##
## setting of SVD
##
######
def my_svd(A) :
#### 
## A = X.Y.Z (not Z.T)
####
    try :
      X, Y ,Z = np.linalg.svd(A)
    except :
      try :
        X, Y, Z = np.linalg.svd(np.dot(np.conj(A.T),A))
        Y = np.sqrt(Y)
        X = np.dot(A, np.conj(Z.T)); X = np.dot(X, np.linalg.inv(np.diag(Y)))
      except :
        try : 
          w,Z = np.linalg.eigh(np.dot(np.conj(A.T),A))
          w = w[::-1]; Z = Z[:,::-1]
          Y = np.sqrt(w)
          X = np.dot(A,Z); X = np.dot(X,np.diag(Y**(-1)))
          Z = np.conj(Z.T) 
        except :
          print('error svd')
          sys.exit()
    return X, Y, Z


def my_svd3(A,q,max_q) :
#### 
## A = X.Y.Z (not Z.T)
####
    try :
#      w, Z = np.linalg.eigh(np.dot(np.conj(A.T),A))
#      w = w[::-1]; Z=Z[:,::-1]
#      Y = np.sqrt(w)
#      invY = Y**(-1)
#      ind = np.where(Y != Y) ; invY[np.min(ind):np.max(ind)] = 0.0 ; Y[np.min(ind):np.max(ind)] = 0.0
#      X = np.dot(np.dot(A,Z),np.diag(invY))
#      X = X[:,0:q]; Y = Y[0:q]; Z = np.conj(Z.T)[0:q,:] 
        w, Z = np.linalg.eigh(np.dot(np.conj(A.T),A))
        w = w[::-1]; Z=Z[:,::-1]
        Y = np.sqrt(w)
        try : 
          ind = np.where(Y != Y)
          Y[np.min(ind):] = float('inf')
        except :
          pass
        X = np.dot(np.dot(A,Z),(np.diag(Y**(-1))))
        X = X[:,0:q]; Y = Y[0:q]; Z = np.conj(Z.T)[0:q,:] 
    except :
      try :
        X, Y, Z = sli.svd(A,max_q)
        X = X[:,0:q]; Y = Y[0:q]; Z = np.conj(Z.T)[0:q,:]
      except : 
        X, Y, Z = np.linalg.svd(A)
        X = X[:,0:q]; Y = Y[0:q]; Z = Z[0:q,:]
    return X, Y, Z
######
##
## make E_matrix
##
######
def calc_Transfer_L(A,O1,O2) :
    V = np.tensordot(np.tensordot(np.tensordot(A,O1,axes=([1,2],[0,1])),np.conj(A),axes=([2,3],[1,2])),O2,axes=([0,2],[0,1]))
    return V

def calc_Transfer_R(A,O1,O2) :
    V = np.tensordot(np.tensordot(np.tensordot(A,O1,axes=([1,2],[0,1])),np.conj(A),axes=([2,3],[1,2])),O2,axes=([1,3],[0,1]))
    return V

def solve_linear_eq(A,B):
    LU = linalg.lu_factor(A)
    x = linalg.lu_solve(LU,B)
    return x

## Left Edge
def make_left_edge_hamiltonian(theta_l,l,SxA,SyA,SzA,SxB,SyB,SzB,J,r,alp,HO,Iee,d,chi):
    temp = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
    temp[4] = np.identity(chi)
    temp[3] = calc_Transfer_L(theta_l,np.reshape(SxB,(d,d,d,d)),temp[4])
    temp[2] = calc_Transfer_L(theta_l,np.reshape(np.imag(SyB),(d,d,d,d)),temp[4])
    temp[1] = calc_Transfer_L(theta_l,np.reshape(SzB,(d,d,d,d)),temp[4])

    cSzz = calc_Transfer_L(theta_l,np.reshape(SzA,(d,d,d,d)),temp[1])
    cSyy =-np.real(calc_Transfer_L(theta_l,np.reshape(np.imag(SyA),(d,d,d,d)),temp[2]))
    cSxx = calc_Transfer_L(theta_l,np.reshape(SxA,(d,d,d,d)),temp[3])
    cH   = calc_Transfer_L(theta_l,np.reshape(HO,(d,d,d,d)),temp[4])
    CL = J * (1.0 - alp) * ( r * ( cSxx + cSyy ) + cSzz ) + cH 

    TI = np.transpose(np.tensordot(np.tensordot(theta_l,np.reshape(Iee,(d,d,d,d)),axes=([1,2],[0,1])),np.conj(theta_l),axes=([2,3],[1,2])),(1,3,0,2))

    I4 = np.identity(chi*chi)
    I2 = np.identity(chi)
    TI = np.reshape(TI,(chi*chi,chi*chi))
    rho_L = np.tensordot(np.diag(l[1,:]),np.diag(np.conj(l[1,:])),axes=(1,0))
    e0 = np.tensordot(CL,rho_L,axes=([0,1],[0,1]))
    print("TEMP Left e0", e0/2)

    B = np.reshape(CL,(chi*chi)) - e0 * np.reshape(I2,(chi*chi))
    H = I4 - TI
    x = solve_linear_eq(H,B)
    temp[0] = np.reshape(x,(chi,chi))
    return temp,e0

## Right Edge
def make_right_edge_hamiltonian(theta_r,l,SxA,SyA,SzA,SxB,SyB,SzB,J,r,alp,HO,Iee,d,chi):
    temp = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
    temp[4] = np.identity(chi)
    temp[3] = calc_Transfer_R(theta_r,np.reshape(SxA,(d,d,d,d)),temp[4])
    temp[2] = calc_Transfer_R(theta_r,np.reshape(np.imag(SyA),(d,d,d,d)),temp[4])
    temp[1] = calc_Transfer_R(theta_r,np.reshape(SzA,(d,d,d,d)),temp[4])

    cSzz = calc_Transfer_R(theta_r,np.reshape(SzB,(d,d,d,d)),temp[1])
    cSyy =-np.real(calc_Transfer_R(theta_r,np.reshape(np.imag(SyB),(d,d,d,d)),temp[2]))
    cSxx = calc_Transfer_R(theta_r,np.reshape(SxB,(d,d,d,d)),temp[3])
    cH   = calc_Transfer_R(theta_r,np.reshape(HO,(d,d,d,d)),temp[4])
    CR = J * (1.0 - alp) * ( r * ( cSxx + cSyy ) + cSzz ) + cH 

    TI = np.transpose(np.tensordot(np.tensordot(theta_r,np.reshape(Iee,(d,d,d,d)),axes=([1,2],[0,1])),np.conj(theta_r),axes=([2,3],[1,2])),(0,2,1,3))

    I4 = np.identity(chi*chi)
    I2 = np.identity(chi)
    TI = np.reshape(TI,(chi*chi,chi*chi))
    rho_R = np.tensordot(np.diag(l[1,:]),np.diag(np.conj(l[1,:])),axes=(1,0))
    e0 = np.tensordot(CR,rho_R,axes=([0,1],[0,1]))
    print("TEMP Right e0", e0/2)

    B = np.reshape(CR,(chi*chi)) - e0 * np.reshape(I2,(chi*chi))
    H = I4 - TI
    x = solve_linear_eq(H,B)
    temp[0] = np.reshape(x,(chi,chi))
    return temp,e0

#################################################
##
##  Real time evolution
##
#################################################
def right_move_contraction(L,S,T,O):
    V = np.tensordot(S, np.tensordot(L, np.tensordot(np.conj(T), O, ([0], [1])), ([1], [0])), ([0, 1], [2, 0]))
    return V

def left_move_contraction(R,S,T,O):
    V = np.tensordot(S, np.tensordot(R, np.tensordot(np.conj(T), O, ([0], [1])), ([1], [1])), ([0, 2], [2, 0]))
    return V


def single_site_contraction(CL,CR,S,T,O):
    val = np.tensordot(
      CL, np.tensordot(
        S, np.tensordot(
            CR, np.tensordot(
                np.conj(T), O, ([0], [1])
            ), ([1], [1])
        ), ([0, 2], [2, 0])
      ), ([0, 1], [0, 1])
    )
    return val 


def calc_corr(S,T,SO,N,d,chi) :
    Crr = np.zeros(N+1,dtype=np.complex128)
    O = np.reshape( np.zeros((N+2)*d*d,dtype=np.complex128),(N+2,d,d) ) 
    for i in range(1,N+1) :
      O[i] = np.identity(d,dtype=np.complex128)

    for i in range(1,N+1) :
      O[i,:,:] = SO.copy()
      CL = np.tensordot(S[0,0,:,:],np.conj(T[0,0,:,:]),axes=(0,0))
      CR = np.tensordot(S[N+1,0,:,:],np.conj(T[N+1,0,:,:]),axes=(1,1))
      for j in range(1,N+1) :
        CL = right_move_contraction(CL,S[j],T[j],O[j])
      val = np.tensordot(CL,CR,axes=([0,1],[0,1]))
      Crr[i] = val
      O[i,:,:] = np.identity(d,dtype=np.complex128)
    return Crr


def calc_corr2(S,T,SO,N,d,chi) :
    Crr = np.zeros(N+1,dtype=np.complex128)
    CL = np.reshape(np.zeros((N+2)*chi*chi,dtype=np.complex128),(N+2,chi,chi))
    CR = np.reshape(np.zeros((N+2)*chi*chi,dtype=np.complex128),(N+2,chi,chi))

    CL[0,:,:]   = np.tensordot(S[0,0,:,:],np.conj(T[0,0,:,:]),axes=(0,0))
    CR[N+1,:,:] = np.tensordot(S[N+1,0,:,:],np.conj(T[N+1,0,:,:]),axes=(1,1))
    for i in range(1,N) :
      CL[i,:,:]     = np.tensordot(S[i,:,:,:],    np.tensordot(CL[i-1,:,:],  np.conj(T[i,:,:,:]),    axes=(1,1)),axes=([0,1],[1,0]))
      CR[N+1-i,:,:] = np.tensordot(S[N+1-i,:,:,:],np.tensordot(CR[N+2-i,:,:],np.conj(T[N+1-i,:,:,:]),axes=(1,2)),axes=([0,2],[1,0]))

    for i in range(1,N+1) :
      val = single_site_contraction(CL[i-1,:,:],CR[i+1,:,:],S[i],T[i],SO) 
      Crr[i] = val
    return Crr

def get_left_edge(S0,S1,SN,L,HL,HLW,T0,T1,TN,M) : 
##############################
# contract_Le.dat
##############################
# (S0*(L*(HL*(S1*(HLW*(((np.conj(T0)*np.conj(M))*np.conj(T1))*(SN*np.conj(TN))))))))
# cpu_cost= 1.4096e+06  memory= 38400
# final_bond_order ()
##############################
    val = np.tensordot(
      S0, np.tensordot(
        L, np.tensordot(
            HL, np.tensordot(
                S1, np.tensordot(
                    HLW, np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.conj(T0), np.conj(M), ([1], [0])
                            ), np.conj(T1), ([1], [1])
                        ), np.tensordot(
                            SN, np.conj(TN), ([1], [1])
                        ), ([2], [1])
                    ), ([2, 3], [0, 1])
                ), ([0, 2], [1, 2])
            ), ([1], [1])
        ), ([1], [1])
      ), ([0, 1], [1, 0])
    )
    return val

def get_right_edge(S0,S1,SN,L,HR,HRW,T0,T1,TN,M):
##############################
# contract_Re.dat
##############################
# (S0*(L*((np.conj(T0)*np.conj(M))*(S1*((SN*HR)*(HRW*(np.conj(T1)*np.conj(TN))))))))
# cpu_cost= 1.4096e+06  memory= 40000
# final_bond_order ()
##############################
    val = np.tensordot(
      S0, np.tensordot(
        L, np.tensordot(
            np.tensordot(
                np.conj(T0), np.conj(M), ([1], [0])
            ), np.tensordot(
                S1, np.tensordot(
                    np.tensordot(
                        SN, HR, ([1], [0])
                    ), np.tensordot(
                        HRW, np.tensordot(
                            np.conj(T1), np.conj(TN), ([2], [0])
                        ), ([2, 3], [0, 2])
                    ), ([1], [1])
                ), ([0, 2], [1, 0])
            ), ([1], [1])
        ), ([1], [1])
      ), ([0, 1], [1, 0])
    )
    return val

def calc_local_ene(S0,S1,S2,SN,L,T0,T1,T2,TN,M,H0):
    val = np.tensordot(
      S0, np.tensordot(
        L, np.tensordot(
            np.tensordot(
                np.conj(T0), np.conj(M), ([1], [0])
            ), np.tensordot(
                S1, np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.conj(T1), np.conj(T2), ([2], [1])
                        ), H0, ([0, 2], [2, 3])
                    ), np.tensordot(
                        S2, np.tensordot(
                            SN, np.conj(TN), ([1], [1])
                        ), ([2], [0])
                    ), ([1, 3], [2, 0])
                ), ([0, 2], [1, 2])
            ), ([1], [1])
        ), ([1], [1])
      ), ([0, 1], [1, 0])
    )
    return val

def calc_energy(S,T,L,M,HL,HLW,h0,h1,HRW,HR,N) :
    val = get_left_edge(S[0,0,:,:],S[1,:,:,:],S[N+1,0,:,:],np.diag(L[0,:]),HL,HLW,T[0,0,:,:],T[1,:,:,:],T[N+1,0,:,:],np.diag(M[0,:]))
    for i in range(1,N) :
      if np.mod(i,2) == 0 :
        res = calc_local_ene(S[0,0,:,:],S[i,:,:,:],S[i+1,:,:,:],S[N+1,0,:,:],np.diag(L[i-1,:]),T[0,0,:,:],T[i,:,:,:],T[i+1,:,:,:],T[N+1,0,:,:],np.diag(M[i-1,:]),h0)
      else :
        res = calc_local_ene(S[0,0,:,:],S[i,:,:,:],S[i+1,:,:,:],S[N+1,0,:,:],np.diag(L[i-1,:]),T[0,0,:,:],T[i,:,:,:],T[i+1,:,:,:],T[N+1,0,:,:],np.diag(M[i-1,:]),h1)
      val += res
    res= get_right_edge(S[0,0,:,:],S[N,:,:,:],S[N+1,0,:,:],np.diag(L[N-1,:]),HR,HRW,T[0,0,:,:],T[N,:,:,:],T[N+1,0,:,:],np.diag(M[N-1,:]))
    val += res 
    return val


def right_move_update(S1,S2,Te,L,d,chi,chi_max) :
    theta_r = np.reshape(np.transpose(np.tensordot(np.tensordot(S1,S2,axes=(2,1)),Te,axes=([0,2],[0,1])),(2,0,3,1)),(d*chi,d*chi))
    P, Q, R = my_svd3(theta_r,chi,chi_max)
    lamb = Q[0:chi]
    V1 = np.reshape(P,(d,chi,chi))
    V2 = np.transpose( np.tensordot(np.diag(lamb),np.reshape(R[0:chi,0:d*chi],(chi,d,chi)),axes=(1,0)),(1,0,2) )
    return V1,V2,lamb

def left_move_update(S1,S2,To,L,d,chi,chi_max) : 
    theta_l = np.reshape(np.transpose( np.tensordot(np.tensordot(S1,S2,axes=(2,1)),To,axes=([0,2],[0,1])),(2,0,3,1)),(d*chi,d*chi))
    P, Q, R = my_svd3(theta_l,chi,chi_max)
    lamb = Q[0:chi]
    V1 = np.tensordot(np.reshape(P,(d,chi,chi)),np.diag(lamb),axes=(2,0))
    V2 = np.transpose(np.reshape(R[0:chi,0:d*chi],(chi,d,chi)),(1,0,2))
    return V1,V2,lamb

def update_left_edge(L0,S1,Tl,Tlw,d,chi,init_norm) :
    temp0 = np.tensordot(L0,S1,axes=(1,1))
    norm0 = np.tensordot(temp0,np.conj(temp0),axes=([0,1,2],[0,1,2]))
    temp1 = np.transpose(np.tensordot(np.tensordot(temp0,Tl,([0],[0])),np.tensordot(Tlw,np.conj(L0),([2],[0])),([0,2],[1,0])),[1,2,0])
    norm1 = np.tensordot(temp1,np.conj(temp1),axes=([0,1,2],[0,1,2]))
    V = temp1 * np.sqrt(norm0/norm1)
    return V

def update_right_edge(SN,R0,Tr,Trw,d,chi,init_norm) :
    temp0 = np.tensordot(SN,R0,axes=(2,0))
    norm0 = np.tensordot(temp0,np.conj(temp0),axes=([0,1,2],[0,1,2]))
    temp1 = np.transpose(np.tensordot(np.tensordot(np.tensordot(temp0,Tr,([2],[0])),Trw,([0,2],[0,1])),np.conj(R0),([2],[0])),[1,0,2])
    norm1 = np.tensordot(temp1,np.conj(temp1),axes=([0,1,2],[0,1,2]))
    V = temp1 * np.sqrt(norm0/norm1)
    return V
#######
##
##  calc correlation
##
#######
def make_initial_MPSs(G,l,d,chi,N):
    S = np.reshape(np.zeros((N+2)*d*chi*chi,dtype=np.complex128),(N+2,d,chi,chi))
    L = np.reshape(np.zeros((N+1)*chi,dtype=np.float64),(N+1,chi))
    S[0,0,0:chi,0:chi]=np.identity(chi)
    S[N+1,0,0:chi,0:chi]=np.identity(chi)
    L[0] = l[1,:].copy()
    for i in range(1,N+1) :
      j = np.mod(i+1,2)
      L[i] = l[j,:].copy()
      S[i,:,:,:] = np.tensordot(G[j,:,:,:],np.diag(L[i,:]),axes=(2,0))
    return S, L

def single_step(S,L,TE,TO,Tl,Tlw,Tr,Trw,chi,chi_max,N,d,init_norm) :
    for i in range(1,N) :
      s = np.mod(i,2)
      V1,V2,l1 = right_move_update(S[i],S[i+1],TE[s],L[i],d,chi,chi_max)
      S[i,:,:,:] = V1; S[i+1,:,:,:] = V2; L[i,:] = l1

#    S[N,:,:,:] = update_right_edge(S[N,:,:,:],S[N+1,0,:,:],Tr,Trw,d,chi,init_norm)
    S[1,:,:,:] = update_left_edge(S[0,0,:,:],S[1,:,:,:],Tl,Tlw,d,chi,init_norm)

    for i in range(N-1,0,-1) :
      s = np.mod(i+1,2)
      V1,V2,l1 = left_move_update(S[i],S[i+1],TO[s],L[i],d,chi,chi_max) 
      S[i,:,:,:] = V1; S[i+1,:,:,:]   = V2; L[i,:] = l1

#    S[1,:,:,:] = update_left_edge(S[0,0,:,:],S[1,:,:,:],Tl,Tlw,d,chi,init_norm)
    S[N,:,:,:] = update_right_edge(S[N,:,:,:],S[N+1,0,:,:],Tr,Trw,d,chi,init_norm)
##
    return S, L
