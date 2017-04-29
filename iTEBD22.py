import scipy.linalg as linalg
import numpy as np
import math
from scipy import integrate
import scipy.linalg.interpolative as sli

import time
import sys
import os
import re

argvs = sys.argv
argc = len(argvs)

if argc == 3 :
  uniform   = float(argvs[1])
  staggered = float(argvs[2])
else : 
  uniform   = 0.0
  staggered = 0.0

eps = 1e-13;
chi = 100; d   = 3;
chi2 = chi
delta = 0.005 ;
i_delta = 0.01j ;
r_range = 100 ;
t_max = 10000 ;
N = 120 ;


def my_svd(A) :
  try :
    w,Z = np.linalg.eigh(np.dot(np.conj(A.T),A))
    w = w[::-1]; Z = Z[:,::-1]
    Y = np.sqrt(w)
    X = np.dot(A,Z); X = np.dot(X,np.diag(Y**(-1)))
  except :
    try :
      X, Y ,Z = np.linalg.svd(A)
    except :
      try : 
        X, Y, Z = np.linalg.svd(np.dot(np.conj(A.T),A))
        Y = np.sqrt(Y)
        X = np.dot(A, Z.T); X = np.dot(X, np.linalg.inv(np.diag(Y)))
      except :
        print 'error'
  return X, Y, Z


def my_svd3(A,q) :
  try :
    X, Y, Z = sli.svd(A,2*q)
    X = X[:,0:q]
    Y = Y[0:q]
    Z = Z[:,0:q]
  except :
    eps = 1e-15
    X, Y, Z = sli.svd(A,eps)
    X = X[:,0:q]
    Y = Y[0:q]
    Z = Z[:,0:q]
  return X, Y, Z


def my_svd2(A) :
  w,Z = np.linalg.eigh(np.dot(A.T ,A))
  w = w[::-1]; Z = Z[:,::-1]
  Y = np.sqrt(w)
  X = np.dot(A,Z); X = np.dot(X,np.diag(Y**(-1)))
  return X, Y, Z



if (d==2) :
  Ie=np.identity(d)

  sz=np.matrix( [[0.5 ,0.0],
                 [0.0,-0.5]] )

  sx=np.matrix( [[0.0 ,1./2.0],
                 [1./2.0 ,0.0]] )

  sy=np.matrix( [[0.0 ,-1.j/2.0],
                 [1.j/2.0 , 0.0]], dtype=np.complex)

  sp=np.matrix( [[0.0 ,1.0],
                 [0.0 ,0.0]] )

elif (d==3) :
  Ie=np.identity(d)

  sz=np.matrix( [[1.0 , 0.0 , 0.0],
                 [0.0 , 0.0 , 0.0],
                 [0.0 , 0.0 ,-1.0]] )

  sx=1./math.sqrt(2.0)*np.matrix( [[0.0 , 1.0 , 0.0],
                                   [1.0 , 0.0 , 1.0],
                                   [0.0 , 1.0 , 0.0]] )

  sy=1./math.sqrt(2.0)*np.matrix( [[0.0 ,-1.j , 0.0],
                                   [1.j , 0.0 ,-1.j],
                                   [0.0 , 1.j , 0.0]] , dtype=np.complex)

  sp=1./math.sqrt(2.0)*np.matrix( [[0.0 , np.sqrt(2.0) , 0.0],
                                   [0.0 , 0.0 , np.sqrt(2.0)],
                                   [0.0 , 0.0 , 0.0]] )
else :
  print 'not supported'


Szz =np.tensordot(sz,sz,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Sxx =np.tensordot(sx,sx,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Syy =np.real(np.tensordot(sy,sy,axes=0).swapaxes(1,2).reshape(d*d,d*d))
Iee =np.tensordot(Ie,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)

SzA =np.tensordot(sz,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SzB =np.tensordot(Ie,sz,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SxA =np.tensordot(sx,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SxB =np.tensordot(Ie,sx,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SyA =np.tensordot(sy,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SyB =np.tensordot(Ie,sy,axes=0).swapaxes(1,2).reshape(d*d,d*d)

#
# Transverse Ising : J=1.0; g=0.5; 
# two-site time evolution operator
#J=1.0; g=0.5
#H0 = J * Szz + g * 0.5 * (SxA + SxB)
#H1 = H0
#
#  Heisenberg Ising ; two-site hamiltonian
J = 1.0; r =1.0; g=0.0; hz=uniform; hst=staggered;
H0 = J * (r*(Sxx+Syy)+Szz) - hz * 0.5 * (SzA + SzB) + hst * 0.5 * (SzA - SzB) - g * 0.5 * (SxA + SxB)
H1 = J * (r*(Sxx+Syy)+Szz) - hz * 0.5 * (SzB + SzA) - hst * 0.5 * (SzB - SzA) - g * 0.5 * (SxB + SxA)

SO = sz
SP = sp
#####
G = np.random.rand(2,d,chi,chi); l = np.random.rand(2,chi)
O = np.reshape(np.zeros(2*d*chi*chi),(2,d,chi,chi))
w0,v0 = np.linalg.eig(H0)
w1,v1 = np.linalg.eig(H1)
Te= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(d,d,d,d) )
To= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(d,d,d,d) )



## Imaginary time evolution
U0= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-delta*(w0))) ) ,np.transpose(v0)) ,(d,d,d,d) )
U1= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-delta*(w1))) ) ,np.transpose(v1)) ,(d,d,d,d) )
U=np.array([U0,U1])
###wavefunc = np.reshape(np.zeros(2*d*chi*d*chi),(2,chi,d,d,chi))
wf0 = np.reshape(np.zeros(d*chi*d*chi),(chi,d,d,chi))
wf1 = np.reshape(np.zeros(d*chi*d*chi),(chi,d,d,chi))
E_GS = np.zeros(4)

start = time.time()
del_e = 1e6;
e_prev= 1e2;
step = 0;
while del_e > eps:
  A = np.mod(step,2); B = np.mod(step+1,2);

  theta = np.tensordot(np.diag(l[B,:]),G[A,:,:,:],axes=(1,1))
  theta = np.tensordot(theta,np.diag(l[A,:],0),axes=(2,0))
  theta = np.tensordot(theta,G[B,:,:,:],axes=(2,1))
  wf1 = theta
  theta = np.tensordot(theta,np.diag(l[B,:],0),axes=(3,0))
  wf0 = theta
#  wavefunc[A] = theta
####
#     chi          chi
#  0 ------------------ 3
#        | theta  |
#        |--------|
#      d |        | d
#        1        2
####
  theta = np.tensordot(theta,U[A],axes=([1,2],[0,1]))
  theta = np.reshape(np.transpose(theta,(2,0,3,1)),(d*chi,d*chi))
  if (np.mod(step,100)==0) :
     print "TEMP E = ", step ,-np.log(np.sum(theta**2))/delta/2

#  X, Y, Z = my_svd3(theta,chi)
  X, Y, Z = my_svd(theta)
  l[A,0:chi] = Y[0:chi]/np.sqrt(sum(Y[0:chi]**2))
  X = np.reshape(X[0:d*chi,0:chi],(d,chi,chi))
  G[A,:,:,:] = np.transpose(np.tensordot(np.diag(l[B,:]**(-1)),X,axes=(1,1)),(1,0,2))
  Z = np.transpose(np.reshape(Z[0:d*chi,0:chi],(d,chi,chi)),(0,2,1))
  G[B,:,:,:] = np.tensordot(Z,np.diag(l[B,:]**(-1)),axes=(2,0)) 
  
  ene = -np.log(np.sum(theta**2))/delta/2
  E_GS[A] = np.real(ene)

  if ((A == 0) & (step > 5000)) :
    del_e = math.fabs(ene - e_prev)
    e_prev = ene 
  
  step += 1
#
#  E = -np.log ( np.tensordot(theta,theta,axes=([0,1,2,3],[0,1,2,3])) ) / delta / 2
#
#  print "R  hz, hst, E = ", hz, hst, -np.log(np.sum(theta**2))/delta/2
#
#
#print "R  hz, hst, E, M = ", hz, hst, -np.log(np.sum(theta**2))/delta/2,(val1+val2)/2.0
print 'TEMP imaginaly time step : ',time.time() - start


start = time.time()
print 'TEMP wf0 ',np.tensordot(wf0,np.conj(wf0),axes=([0,1,2,3],[0,1,2,3]))
Iec = np.diag(np.ones(chi))
tt = np.tensordot(Iec,np.tensordot(np.tensordot(np.tensordot(wf1,wf1,axes=(3,0)),np.diag(l[1,]),axes=(5,0)),Iec,axes=(5,0)),axes=(1,0))
print 'TEMP wf1+wf1+lb', np.tensordot(tt,np.conj(tt),axes=([0,1,2,3,4,5],[0,1,2,3,4,5]))
print 'TEMP ', time.time() - start



start = time.time()
######
#
# make E_matrix
#
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
def make_left_edge_hamiltonian(theta_l,SxA,SyA,SzA,SxB,SyB,SzB,d,chi):
  temp = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
  temp[4] = np.identity(chi)
  temp[3] = calc_Transfer_L(theta_l,np.reshape(SxB,(d,d,d,d)),temp[4])
  temp[2] = calc_Transfer_L(theta_l,np.reshape(np.imag(SyB),(d,d,d,d)),temp[4])
  temp[1] = calc_Transfer_L(theta_l,np.reshape(SzB,(d,d,d,d)),temp[4])

  cSzz = calc_Transfer_L(theta_l,np.reshape(SzA,(d,d,d,d)),temp[1])
  cSyy = np.real(calc_Transfer_L(theta_l,np.reshape(-np.imag(SyA),(d,d,d,d)),temp[2]))
  cSxx = calc_Transfer_L(theta_l,np.reshape(SxA,(d,d,d,d)),temp[3])
  cH   = calc_Transfer_L(theta_l,np.reshape(H0,(d,d,d,d)),temp[4])
  CL = ( cSxx + cSyy + cSxx + cH )

  TI = np.transpose(np.tensordot(np.tensordot(theta_l,np.reshape(Iee,(d,d,d,d)),axes=([1,2],[0,1])),np.conj(theta_l),axes=([2,3],[1,2])),(0,2,1,3))

  I4 = np.identity(chi*chi)
  I2 = np.identity(chi)
  TI = np.reshape(TI,(chi*chi,chi*chi))
  rho_L = np.tensordot(np.diag(l[1,:]),np.diag(np.conj(l[1,:])),axes=(1,0))
  e0 = np.tensordot(CL,rho_L,axes=([0,1],[0,1]))
  print "TEMP Left e0", e0/2

  B = np.reshape(CL,(chi*chi)) - e0/2 * np.reshape(I2,(chi*chi))
  H = I4 - TI
  x = solve_linear_eq(H,B)
  temp[0] = np.reshape(x,(chi,chi))
  return temp,e0

## Right Edge
def make_right_edge_hamiltonian(theta_r,SxA,SyA,SzA,SxB,SyB,SzB,d,chi):
  temp = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
  temp[4] = np.identity(chi)
  temp[3] = calc_Transfer_R(theta_r,np.reshape(SxA,(d,d,d,d)),temp[4])
  temp[2] = calc_Transfer_R(theta_r,np.reshape(np.imag(SyA),(d,d,d,d)),temp[4])
  temp[1] = calc_Transfer_R(theta_r,np.reshape(SzA,(d,d,d,d)),temp[4])

  cSzz = calc_Transfer_R(theta_r,np.reshape(SzB,(d,d,d,d)),temp[1])
  cSyy = np.real(calc_Transfer_R(theta_r,np.reshape(-np.imag(SyB),(d,d,d,d)),temp[2]))
  cSxx = calc_Transfer_R(theta_r,np.reshape(SxB,(d,d,d,d)),temp[3])
  cH   = calc_Transfer_R(theta_r,np.reshape(H0,(d,d,d,d)),temp[4])
  CR = ( cSxx + cSyy + cSxx + cH )

  TI = np.transpose(np.tensordot(np.tensordot(theta_r,np.reshape(Iee,(d,d,d,d)),axes=([1,2],[0,1])),np.conj(theta_r),axes=([2,3],[1,2])),(1,3,0,2))

  I4 = np.identity(chi*chi)
  I2 = np.identity(chi)
  TI = np.reshape(TI,(chi*chi,chi*chi))
  rho_R = np.tensordot(np.diag(l[1,:]),np.diag(np.conj(l[1,:])),axes=(1,0))
  e0 = np.tensordot(CR,rho_R,axes=([0,1],[0,1]))
  print "TEMP Right e0", e0/2

  B = np.reshape(CR,(chi*chi)) - e0/2 * np.reshape(I2,(chi*chi))
  H = I4 - TI
  x = solve_linear_eq(H,B)
  temp[0] = np.reshape(x,(chi,chi))
  return temp,e0

#############
EL = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
ER = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
#
#  0 -lb - Ga -- Gb -- 3       |-- Ga -- Gb -- 2
#          |     |             |   |     |
#          | 1   | 2           |   | 0   | 1
#                              |             Tf
#             theta            |   | 3   | 4
#                              |   |     |
#                              |-- Ga*-- Gb*-- 5
#
theta_l = np.tensordot(np.diag(l[1,:]),G[0,:,:,:],axes=(1,1))
theta_l = np.tensordot(theta_l,np.diag(l[0,:]),axes=(2,0))
theta_l = np.tensordot(theta_l,G[1,:,:,:],axes=(2,1))

theta_r = np.tensordot(G[0,:,:,:],np.diag(l[0,:]),axes=(2,0))
theta_r = np.tensordot(theta_r,G[1,:,:,:],axes=(2,1))
theta_r = np.transpose(np.tensordot(theta_r,np.diag(l[1,:]),axes=(3,0)),(1,0,2,3))

EL, e0 = make_left_edge_hamiltonian(theta_l,SxA,SyA,SzA,SxB,SyB,SzB,d,chi)
E_GS[2] = e0
ER, e0 = make_right_edge_hamiltonian(theta_r,SxA,SyA,SzA,SxB,SyB,SzB,d,chi)
E_GS[3] = e0

Hl=EL[0].copy()
Hr=ER[0].copy()
w0,v0 = np.linalg.eig(EL[0])
w1,v1 = np.linalg.eig(ER[0])
Tl= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(chi,chi) )
Tr= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(chi,chi) )

Ief = np.diag(np.ones(chi))
effl_Szz =np.tensordot(EL[3],sz,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
effl_Sxx =np.tensordot(EL[1],sx,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
effl_Syy =np.tensordot(EL[2],-np.imag(sy),axes=0).swapaxes(1,2).reshape(chi*d,chi*d)

effl_SxA =np.tensordot(EL[1],Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
effl_SxB =np.tensordot(Ief,sx,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
##effl_SyA =np.tensordot(-EL[2]*1e0j,Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
##effl_SyB =np.tensordot(Ief,sy,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
effl_SzA =np.tensordot(EL[3],Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
effl_SzB =np.tensordot(Ief,sz,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)

effr_Szz =np.tensordot(sz,ER[1],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
effr_Sxx =np.tensordot(sx,ER[3],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
effr_Syy =np.tensordot(-np.imag(sy),ER[2],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)

effr_SxA =np.tensordot(Ie,ER[3],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
effr_SxB =np.tensordot(sx,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
##effr_SyA =np.tensordot(Ie,-ER[2]*1e0j,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
##effr_SyB =np.tensordot(sy,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
effr_SzA =np.tensordot(Ie,ER[1],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
effr_SzB =np.tensordot(sz,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)

Hlw = J * (r*(effl_Sxx+effl_Syy)+effl_Szz) - hz * 0.5 * (effl_SzA + effl_SzB) - hst * 0.5 * (effl_SzA - effl_SzB) - g * 0.5 * (effl_SxA + effl_SxB)  # B-A  
Hrw = J * (r*(effr_Sxx+effr_Syy)+effr_Szz) - hz * 0.5 * (effr_SzA + effr_SzB) + hst * 0.5 * (effr_SzA - effr_SzB) - g * 0.5 * (effr_SxA + effr_SxB)  # A-B
w0,v0 = np.linalg.eig(Hlw)
w1,v1 = np.linalg.eig(Hrw)
Tlw= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(chi,d,chi,d) )
Trw= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(d,chi,d,chi) )

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
  Crr = np.zeros(N+1,dtype=np.complex64)
  O = np.reshape( np.zeros((N+2)*d*d,dtype=np.complex64),(N+2,d,d) ) 
  for i in range(1,N+1) :
    O[i] = np.identity(d,dtype=np.complex64)

  for i in range(1,N+1) :
    O[i,:,:] = SO.copy()
    CL = np.tensordot(S[0,0,:,:],np.conj(T[0,0,:,:]),axes=(0,0))
    CR = np.tensordot(S[N+1,0,:,:],np.conj(T[N+1,0,:,:]),axes=(1,1))
    for j in range(1,N+1) :
      CL = right_move_contraction(CL,S[j],T[j],O[j])
    val = np.tensordot(CL,CR,axes=([0,1],[0,1]))
    Crr[i] = val
    O[i,:,:] = np.identity(d,dtype=np.complex64)
  return Crr


def calc_corr2(S,T,SO,N,d,chi) :
  Crr = np.zeros(N+1,dtype=np.complex64)
  CL = np.reshape(np.zeros((N+2)*chi*chi,dtype=np.complex64),(N+2,chi,chi))
  CR = np.reshape(np.zeros((N+2)*chi*chi,dtype=np.complex64),(N+2,chi,chi))

  CL[0,:,:]   = np.tensordot(S[0,0,:,:],np.conj(T[0,0,:,:]),axes=(0,0))
  CR[N+1,:,:] = np.tensordot(S[N+1,0,:,:],np.conj(T[N+1,0,:,:]),axes=(1,1))
  for i in range(1,N) :
    CL[i,:,:]     = np.tensordot(S[i,:,:,:],    np.tensordot(CL[i-1,:,:],  np.conj(T[i,:,:,:]),    axes=(1,1)),axes=([0,1],[1,0]))
    CR[N+1-i,:,:] = np.tensordot(S[N+1-i,:,:,:],np.tensordot(CR[N+2-1,:,:],np.conj(T[N+1-i,:,:,:]),axes=(1,2)),axes=([0,2],[1,0]))
    print "L,R", i,N+1-i    

  for i in range(1,N+1) :
    val = single_site_contraction(CL[i-1,:,:],CR[i+1,:,:],S[i],T[i],SO) 
    Crr[i] = val
  return Crr

def contract_H0(CL,H0,S,T) :
##############################
# contract_H0.dat
##############################
# ((np.conj(T)*(CL*H0))*S)
# cpu_cost= 537600  memory= 19216
# final_bond_order (f, e, i, h)
##############################
  V = np.transpose(
    np.tensordot(
        np.tensordot(
            np.conj(T), np.tensordot(
                CL, H0, ([1, 2], [0, 2])
            ), ([0, 1], [3, 1])
        ), S, ([1], [1])
    ), [3, 2, 1, 0]
  )
  return V

def contract_H1(CL,H1,S,T) :
##############################
# contract_H1.dat
##############################
# ((S*(CL*H1))*T)
# cpu_cost= 537600  memory= 19216
# final_bond_order (f, i, g, h)
##############################
  V = np.tensordot(
    np.tensordot(
        S, np.tensordot(
            CL, H1, ([1, 2], [0, 2])
        ), ([0, 1], [2, 0])
    ), np.conj(T), ([1], [1])
  )
  return V

def get_left_edge(S0,S1,HL,HLW,T0,T1) : 
##############################
# contract_Le.dat
##############################
# ((((S0*HL)*S1)*HLW)*(np.conj(T0)*np.conj(T1)))
# cpu_cost= 832000  memory= 20800
# final_bond_order (d, g, i, j)
##############################
  V = np.tensordot(
    np.tensordot(
        np.tensordot(
            np.tensordot(
                S0, HL, ([0], [0])
            ), S1, ([0], [1])
        ), HLW, ([0, 1], [0, 1])
    ), np.tensordot(
        np.conj(T0), np.conj(T1), ([1], [1])
    ), ([1], [0])
  )
  return V

def get_right_edge(CL,HR,HRW,SE,TE):
##############################
# contract_Re.dat
##############################
# ((SE*HR)*(CL*(np.conj(TE)*HRW)))
# cpu_cost= 577600  memory= 24000
# final_bond_order ()
##############################
  val = np.tensordot(
      np.tensordot(
        SE, HR, ([1], [0])
      ), np.tensordot(
        CL, np.tensordot(
            np.conj(TE), HRW, ([0], [3])
        ), ([1, 2, 3], [1, 3, 0])
      ), ([0, 1], [0, 1])
  )
  return val

def calc_energy(S,T,HL,HLW,h0,h1,HRW,HR) :
  CL = get_left_edge(S[0,0,:,:],S[1,:,:,:],HL,HLW,T[0,0,:,:],T[1,:,:,:])
  for i in range(2,N+1) :
    if np.mod(i,2) == 0 :
      CL = contract_H0(CL,h0,S[i,:,:,:],T[i,:,:,:])
    else :
      CL = contract_H1(CL,h1,S[i,:,:,:],T[i,:,:,:])
  val = get_right_edge(CL,HR,HRW,S[N+1,0,:,:],T[N+1,0,:,:])
  return val


def right_move_update(S1,S2,Te,L,d,chi) :
    theta_r = np.tensordot(S1,S2,axes=(2,1))
    theta_r = np.reshape(np.transpose(np.tensordot(theta_r,Te,axes=([0,2],[0,1])),(2,0,3,1)),(d*chi,d*chi))
    P, Q, R = my_svd3(theta_r,chi)
    lamb = Q[0:chi]
##    lamb = Q[0:chi]/np.sqrt(sum(Q[0:chi])**2)
##    lamb = Q[0:chi]/np.sqrt(sum(Q[0:chi]**2))
    V1 = np.reshape(P,(d,chi,chi))
    V2 = np.transpose( np.tensordot(np.diag(lamb),np.reshape(R[0:d*chi,0:chi],(d,chi,chi)),axes=(1,2)),(1,0,2) )
    return V1,V2,lamb

def left_move_update(S1,S2,To,L,d,chi) : 
    theta_l = np.tensordot(np.tensordot(S1,S2,axes=(2,1)),To,axes=([0,2],[0,1]))
    theta_l = np.reshape(np.transpose(theta_l,(2,0,3,1)),(d*chi,d*chi))
    P, Q, R = my_svd3(theta_l,chi)
##    lamb = Q[0:chi]/np.sqrt(sum(Q[0:chi])**2)
##    lamb = Q[0:chi]/np.sqrt(sum(Q[0:chi]**2))
    lamb = Q[0:chi]
    V1 = np.tensordot(np.reshape(P,(d,chi,chi)),np.diag(lamb),axes=(2,0))
    V2 = np.transpose(np.reshape(R[0:d*chi,0:chi],(d,chi,chi)),(0,2,1))
    return V1,V2,lamb

def update_left_edge(L0,S1,Tl,Tlw,d,chi,init_norm) :
  temp0 = np.tensordot(L0,S1,axes=(1,1))
  norm0 = np.tensordot(temp0,np.conj(temp0),axes=([0,1,2],[0,1,2]))
  temp1 = np.transpose(np.tensordot(np.tensordot(temp0,Tl,([0],[0])),np.tensordot(Tlw,np.conj(L0),([2],[0])),([0,2],[1,0])),[1,2,0])
  V = np.conj(temp1)

  temp2 = np.tensordot(L0,V,axes=(1,1))
  norm2 = np.tensordot(temp2,np.conj(temp2),axes=([0,1,2],[0,1,2]))
  V = V * np.sqrt(norm0/norm2)
  print 'TEMP L norm check : ',norm0,norm2
  return V

def update_right_edge(SN,R0,Tr,Trw,d,chi,init_norm) :
  temp0 = np.tensordot(SN,R0,axes=(2,0))
  norm0 = np.tensordot(temp0,np.conj(temp0),axes=([0,1,2],[0,1,2]))
  temp1 = np.transpose(np.tensordot(np.tensordot(np.tensordot(temp0,Tr,([2],[0])),Trw,([0,2],[0,1])),np.conj(R0),([2],[0])),[1,0,2])
  V = np.conj(temp1)

  temp2 = np.tensordot(V,R0,axes=(2,0))
  norm2 = np.tensordot(temp2,np.conj(temp2),axes=([0,1,2],[0,1,2]))
  V = V * np.sqrt(norm0/norm2)
  print 'TEMP R norm check :',norm0,norm2
  return V

def update_left_edge2(L0,S1,Tl,Tlw,d,chi,init_norm) :
  norm0 = np.tensordot(L0,np.conj(L0),axes=([0,1],[0,1]))
  temp0 = np.tensordot(L0,S1,axes=(1,1))
  temp0 = np.tensordot(np.tensordot(Tl,temp0,axes=(1,0)),Tlw,axes=([0,1],[0,1]))
  temp0 = np.tensordot(np.conj(S1),temp0,axes=([0,2],[2,0]))
  norm1 = np.tensordot(temp0,np.conj(temp0),axes=([0,1],[0,1]))
  V = (L0) * np.sqrt(np.sqrt(np.conj(norm1)*norm1)) 
  print 'TEMP L norm check :',norm0,norm1
#  V = np.conj(temp0.T)
  return V

def update_right_edge2(SN,R0,Tr,Trw,d,chi,init_norm) :
  norm0 = np.tensordot(R0,np.conj(R0),axes=([0,1],[0,1]))
  temp0 = np.tensordot(SN,R0,axes=(2,0))
  temp0 = np.tensordot(np.tensordot(temp0,Tr,axes=(2,0)),Trw,axes=([0,2],[0,1]))
  temp0 = np.tensordot(np.conj(SN),temp0,axes=([0,1],[1,0]))
  norm1 = np.tensordot(temp0,np.conj(temp0),axes=([0,1],[0,1]))
  V = (R0) * np.sqrt(np.sqrt(np.conj(norm1)*norm1)) 
  print 'TEMP R norm check :',norm0,norm1
#  V = np.conj(temp0)
  return V
##
print 'TEMP  ',time.time() - start


start = time.time()
#######
#
#  calc correlation
#
#######
def make_initial_MPSs(G,l,d,chi,N):
  S = np.reshape(np.zeros((N+2)*d*chi*chi,dtype=np.complex64),(N+2,d,chi,chi))
  L = np.reshape(np.zeros((N+1)*chi,dtype=np.float64),(N+1,chi))
  S[0,0,0:chi,0:chi]=np.identity(chi)
  S[N+1,0,0:chi,0:chi]=np.identity(chi)
  L[0] = l[1,:].copy()
  for i in range(1,N+1) :
    j = np.mod(i+1,2)
    L[i] = l[j,:].copy()
    S[i,:,:,:] = np.tensordot(G[j,:,:,:],np.diag(L[i,:]),axes=(2,0))
  return S, L

S, L = make_initial_MPSs(G,l,d,chi,N)
T, M = make_initial_MPSs(G,l,d,chi,N)

S[1,:,:,:] = np.transpose(np.tensordot(np.diag(L[0,:]),S[1,:,:,:],axes=(1,1)),(1,0,2))
T[1,:,:,:] = np.transpose(np.tensordot(np.diag(M[0,:]),T[1,:,:,:],axes=(1,1)),(1,0,2))

##
h0 = np.reshape(np.identity(d*d),(d,d,d,d))
h1 = np.reshape(np.identity(d*d),(d,d,d,d))
Hl = np.reshape(np.identity(chi),(chi,chi))
Hr = np.reshape(np.identity(chi),(chi,chi))
Hlw = np.reshape(np.identity(chi*d),(chi,d,chi,d))
Hrw = np.reshape(np.identity(chi*d),(d,chi,d,chi))
norm = calc_energy(S,S,Hl,Hlw,h0,h1,Hrw,Hr)
print 'TEMP norm check', norm
##
#h0 = np.reshape(H0,(d,d,d,d))
#h1 = np.reshape(H1,(d,d,d,d))
#Hl = np.reshape(Hl,(chi,chi))
#Hr = np.reshape(Hr,(chi,chi))
#Hlw = np.reshape(Hlw,(chi,d,chi,d))
#Hrw = np.reshape(Hrw,(d,chi,d,chi))
#E_total = calc_energy(S,S,Hl,Hlw,h0,h1,Hrw,Hr)
#print 'TEMP E total ', E_total
##




S[N/2,:,:,:] = np.tensordot(SP,S[N/2,:,:,:],axes=(0,0))
#Corr = np.zeros(N+1,dtype=np.complex64)
#Corr = calc_corr(S,T,np.conj(SP.T),N,d,chi)
Mag = np.zeros(N+1,dtype=np.complex64)
Mag = calc_corr(S,S,sz,N,d,chi)

for i in range(1,N+1):
  print "R 0",i-N/2,np.real(Mag[i])
print "R  "

Et = np.reshape(Iee,(d,d,d,d))

init_norm = norm
E_total = (E_GS[0] / 2 + E_GS[1] / 2) * N / 2 + (E_GS[2] / 2 + E_GS[3] / 2)

print time.time() - start


start = time.time()
for t_step in range(1,t_max) :
    t = np.imag(i_delta) * np.float(t_step)
    print 'TEMP t=',t
    start = time.time()
    factor = math.cos(t*E_total) + math.sin(t*E_total) * 1.0j

    TR = [Et,Te]
    for i in range(1,N) :
      s = np.mod(i,2)
      V1,V2,l1 = right_move_update(S[i],S[i+1],TR[s],L[i],d,chi)
      S[i,:,:,:] = V1; S[i+1,:,:,:] = V2; L[i,:] = l1

    TL = [Et,To]
    for i in range(N-1,0,-1) :
      s = np.mod(i+1,2)
      V1,V2,l1 = left_move_update(S[i],S[i+1],TL[s],L[i],d,chi) 
      S[i,:,:,:] = V1; S[i+1,:,:,:]   = V2; L[i,:] = l1

    S[1,:,:,:] = update_left_edge(S[0,0,:,:],S[1,:,:,:],Tl,Tlw,d,chi,init_norm)
    S[N,:,:,:] = np.transpose(np.tensordot(np.diag(L[N-1,:]),S[N,:,:,:],axes=(1,1)),(1,0,2))
    S[N,:,:,:] = update_right_edge(S[N,:,:,:],S[N+1,0,:,:],Tr,Trw,d,chi,init_norm)
    S[N,:,:,:] = np.transpose(np.tensordot(np.diag(L[N-1,:]**(-1)),S[N,:,:,:],axes=(1,1)),(1,0,2))

    now = time.time()
    print 'TEMP single evolusion time =',now - start
    start = time.time()
    norm = calc_energy(S,S,Hl,Hlw,h0,h1,Hrw,Hr)
    print 'TEMP norm check ', norm
    init_norm = norm
    now = time.time()
    print 'TEMP energy evaluation ',now - start

    start = time.time()
#    Corr = calc_corr2(S,T,SO,N,d,chi)
    Mag = calc_corr(S,S,sz,N,d,chi)
    now = time.time()
    print 'TEMP Corr :',now - start
    if (np.mod(t_step,1) == 0) :
      for i in range(1,N+1):
        print "R ",t,i-N/2,np.real(Mag[i])
      print "R  "

