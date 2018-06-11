import scipy.linalg as linalg
import numpy as np
import math
from scipy import integrate
import scipy.linalg.interpolative as sli

import time
import sys
import os
import re
import gc

import iTEBD_funcs as mf


argvs = sys.argv
argc = len(argvs)
eps = 1e-10
t_start = 1

if argc == 4 :
  uniform   = float(argvs[1])
  staggered = float(argvs[2])
  read_dump_file = False
  if( str(argvs[3]) == 'T' ) :
    read_dump_file = True
else : 
  uniform   = 0.0
  staggered = 0.0
  read_dump_file = False
######
##
##  d: dimension of spin(S=1/2:2, S=1:3)
##  N: number of sites
##  t_max : real time steps
##  delta : width of imaginaly time evolution 
##  i_delta : width of real time evolution
##
######
chi = 60; d   = 3;
chi_max = int(chi * 1.5)
delta = 0.005 ;
N = 200 ;
i_delta = 0.01j  ;
t_max = 20000 ;
J = 1.0; alp = 0.0 ; r =1.0; g=0.0 ; Dz = 0.0 ; hz=uniform; hst=staggered;
######
#
if(read_dump_file) :
  chi,d,chi_max,N,i_delta,t_start,J,alp,Dz,r,hz,hst,g = mf.unpack_parameters(read_dump_file)
#
######
##
##  definition of spins
##
######
mf.print_sim_env(chi,d,N,i_delta,chi_max,J,alp,Dz,r,hz,hst,g) 
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
  print('not supported')

file_name0='corr.z.txt'
file_name1='corr.txt'
SO = sy
SP = sz

Szz =np.tensordot(sz,sz,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Sxx =np.tensordot(sx,sx,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Syy =np.real(np.tensordot(sy,sy,axes=0).swapaxes(1,2).reshape(d*d,d*d))
Iee =np.tensordot(Ie,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)

#######
##
##  Hamiltonian
##
#######
SzA =np.tensordot(sz,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SzB =np.tensordot(Ie,sz,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SxA =np.tensordot(sx,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SxB =np.tensordot(Ie,sx,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SyA =np.tensordot(sy,Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
SyB =np.tensordot(Ie,sy,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Sz2A =np.tensordot(np.dot(sz,sz),Ie,axes=0).swapaxes(1,2).reshape(d*d,d*d)
Sz2B =np.tensordot(Ie,np.dot(sz,sz),axes=0).swapaxes(1,2).reshape(d*d,d*d)
#  Heisenberg Ising ; two-site hamiltonian
H0 = J * (1.0 + alp) * (r*(Sxx+Syy)+Szz) - hz * 0.5 * (SzA + SzB) + hst * 0.5 * (SzB - SzA) - g * 0.5 * (SxA + SxB) + Dz * 0.5 * (Sz2A + Sz2B)
## A<->B B<->A 
H1 = J * (1.0 - alp) * (r*(Sxx+Syy)+Szz) - hz * 0.5 * (SzB + SzA) - hst * 0.5 * (SzB - SzA) - g * 0.5 * (SxB + SxA) + Dz * 0.5 * (Sz2B + Sz2A)


#####
G = np.random.rand(2,d,chi,chi); l = np.random.rand(2,chi)
#G = np.reshape(np.ones(2*d*chi*chi),(2,d,chi,chi)); l = np.reshape(np.ones(2*chi),(2,chi))
O = np.reshape(np.zeros(2*d*chi*chi),(2,d,chi,chi))
w0,v0 = np.linalg.eig(H0)
w1,v1 = np.linalg.eig(H1)
Te= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(d,d,d,d) )
To= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(d,d,d,d) )
######
##
## Imaginary time evolution
##
######
U0= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-delta*(w0))) ) ,np.transpose(v0)) ,(d,d,d,d) )
U1= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-delta*(w1))) ) ,np.transpose(v1)) ,(d,d,d,d) )
U=np.array([U0,U1])
###wavefunc = np.reshape(np.zeros(2*d*chi*d*chi),(2,chi,d,d,chi))
wf0 = np.reshape(np.zeros(d*chi*d*chi),(chi,d,d,chi))
E_GS = np.zeros(4)

start = time.time()
del_e = 1e6;
e_prev= 1e2;
step = 0;


if(read_dump_file) :
  R,K,S,L,T,M,TE,TO,Tl,Tlw,Tr,Trw = mf.read_dump()
else :
  while del_e > eps:
    A = np.mod(step,2); B = np.mod(step+1,2);

    theta = np.tensordot(np.diag(l[B,:]),G[A,:,:,:],axes=(1,1))
    theta = np.tensordot(theta,np.diag(l[A,:],0),axes=(2,0))
    theta = np.tensordot(theta,G[B,:,:,:],axes=(2,1))
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

    X, Y, Z = mf.my_svd3(theta,chi,chi_max)
#  X, Y, Z = mf.my_svd(theta)
    l[A,:] = Y/np.sqrt(sum(Y**2))
    X = np.reshape(X,(d,chi,chi))
    G[A,:,:,:] = np.transpose(np.tensordot(np.diag(l[B,:]**(-1)),X,axes=(1,1)),(1,0,2))
    Z = np.transpose(np.reshape(Z,(chi,d,chi)),(1,0,2))
    G[B,:,:,:] = np.tensordot(Z,np.diag(l[B,:]**(-1)),axes=(2,0)) 
  
    if (np.mod(step,100)==0) :
       print("TEMP E = ", step ,E_GS[0],E_GS[1])

    ene = -np.log(np.sum(theta**2))/delta/2
    E_GS[A] = np.real(ene)

    if ((A == 0) & (step > 10000)) :
      del_e = math.fabs(ene - e_prev)
      e_prev = ene 
  
    step += 1
#
  print('TEMP imaginaly time step : ',time.time() - start)
  print('TEMP wf0 ',np.tensordot(wf0,np.conj(wf0),axes=([0,1,2,3],[0,1,2,3])))


  start = time.time()
######
##
## make E_matrix
##
######
#############
  EL = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
  ER = np.reshape(np.zeros(5*chi*chi),(5,chi,chi))
  H2 = J * (1.0 + alp) * (r*(Sxx+Syy)+Szz) - hz * (SzA + SzB) + hst * (SzB - SzA) - g * (SxA + SxB) + Dz * (Sz2A + Sz2B)
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

  EL, e0 = mf.make_left_edge_hamiltonian(theta_l,l,SxA,SyA,SzA,SxB,SyB,SzB,J,r,alp,H2,Iee,d,chi)
  E_GS[2] = e0
  ER, e0 = mf.make_right_edge_hamiltonian(theta_r,l,SxA,SyA,SzA,SxB,SyB,SzB,J,r,alp,H2,Iee,d,chi)
  E_GS[3] = e0

  Hl=EL[0].copy()
  Hr=ER[0].copy()
  w0,v0 = np.linalg.eig(EL[0])
  w1,v1 = np.linalg.eig(ER[0])
  Tl= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(chi,chi) )
  Tr= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(chi,chi) )
##
#  A and B mean A site and B site!!(definition is different.)
##
  Ief = np.diag(np.ones(chi))
  effl_Szz =np.tensordot(EL[1],sz,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_Sxx =np.tensordot(EL[3],sx,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_Syy =np.tensordot(EL[2],-np.imag(sy),axes=0).swapaxes(1,2).reshape(chi*d,chi*d)

  effl_SxB =np.tensordot(EL[3],Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_SxA =np.tensordot(Ief,sx,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
##  effl_SyB =np.tensordot(-EL[2]*1e0j,Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
##  effl_SyA =np.tensordot(Ief,sy,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_SzB =np.tensordot(EL[1],Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_SzA =np.tensordot(Ief,sz,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_Sz2B =np.tensordot(np.dot(EL[1],EL[1]),Ie,axes=0).swapaxes(1,2).reshape(chi*d,chi*d)
  effl_Sz2A =np.tensordot(Ief,np.dot(sz,sz),axes=0).swapaxes(1,2).reshape(chi*d,chi*d)

  effr_Szz =np.tensordot(sz,ER[1],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_Sxx =np.tensordot(sx,ER[3],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_Syy =np.tensordot(-np.imag(sy),ER[2],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)

  effr_SxA =np.tensordot(Ie,ER[3],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_SxB =np.tensordot(sx,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
##  effr_SyA =np.tensordot(Ie,-ER[2]*1e0j,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
##  effr_SyB =np.tensordot(sy,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_SzA =np.tensordot(Ie,ER[1],axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_SzB =np.tensordot(sz,Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_Sz2A =np.tensordot(Ie,np.dot(ER[1],ER[1]),axes=0).swapaxes(1,2).reshape(d*chi,d*chi)
  effr_Sz2B =np.tensordot(np.dot(sz,sz),Ief,axes=0).swapaxes(1,2).reshape(d*chi,d*chi)

#  Hlw = J * (r*(effl_Sxx+effl_Syy)+effl_Szz) - hz * 0.5 * (effl_SzA + effl_SzB) + hst * 0.5 * (effl_SzB - effl_SzA) - g * 0.5 * (effl_SxA + effl_SxB)  # B-A  
#  Hrw = J * (r*(effr_Sxx+effr_Syy)+effr_Szz) - hz * 0.5 * (effr_SzA + effr_SzB) + hst * 0.5 * (effr_SzB - effr_SzA) - g * 0.5 * (effr_SxA + effr_SxB)  # B-A
  Hlw = J * (1.0 - alp) * (r*(effl_Sxx+effl_Syy)+effl_Szz) - hz * 0.5 *( effl_SzA) - hst * 0.5 * (effl_SzA) - g * 0.5 * (effl_SxA) + Dz * 0.5 * (effl_Sz2A) # B-A  
  Hrw = J * (1.0 - alp) * (r*(effr_Sxx+effr_Syy)+effr_Szz) - hz * 0.5 * (effr_SzB) + hst * 0.5 * (effr_SzB) - g * 0.5 * (effr_SxB) + Dz * 0.5 * (effr_Sz2B) # B-A
  w0,v0 = np.linalg.eig(Hlw)
  w1,v1 = np.linalg.eig(Hrw)
  Tlw= np.reshape( np.dot( np.dot( v0,np.diag(np.exp(-i_delta*(w0))) ) ,np.transpose(v0)) ,(chi,d,chi,d) )
  Trw= np.reshape( np.dot( np.dot( v1,np.diag(np.exp(-i_delta*(w1))) ) ,np.transpose(v1)) ,(d,chi,d,chi) )

#################################################
##
##  Real time evolution
##
#################################################
##
  print('TEMP cost for making edge tensors :',time.time() - start)
  start = time.time()
#######
##
##  calc correlation
##
#######
  Et = np.reshape(np.identity(d*d),(d,d,d,d))
  TE = [Et,Te]
  TO = [Et,To]
  TE2 = [Et,Et]
  TO2 = [Et,Et]
  R, K = mf.make_initial_MPSs(G,l,d,chi,N)
  S, L = mf.make_initial_MPSs(G,l,d,chi,N)
  T, M = mf.make_initial_MPSs(G,l,d,chi,N)
##
#  h0 = np.reshape(np.identity(d*d),(d,d,d,d))
#  h1 = np.reshape(np.identity(d*d),(d,d,d,d))
  Hl = np.identity(chi)
  Hr = np.identity(chi)
#  Hlw = np.reshape(np.identity(chi*d),(chi,d,chi,d))
#  Hrw = np.reshape(np.identity(chi*d),(d,chi,d,chi))
  h0 = np.reshape(H0,(d,d,d,d))
  h1 = np.reshape(H1,(d,d,d,d))
#  Hl = np.reshape(Hl,(chi,chi))
#  Hr = np.reshape(Hr,(chi,chi))
  Hlw = np.reshape(Hlw,(chi,d,chi,d))
  Hrw = np.reshape(Hrw,(d,chi,d,chi))
  E_total = mf.calc_energy(S,S,L,L,Hl,Hlw,h0,h1,Hrw,Hr,N)
  print('TEMP Energy per site (averaged) :', E_total / (N+2))
  E_total2 = (E_GS[0] / 2 + E_GS[1] / 2) * N / 2 + (E_GS[2] / 2 + E_GS[3] / 2)
##
##
  h0 = np.reshape(np.identity(d*d),(d,d,d,d))
  h1 = np.reshape(np.identity(d*d),(d,d,d,d))
  Hl = np.reshape(np.identity(chi),(chi,chi))
  Hr = np.reshape(np.identity(chi),(chi,chi))
  Hlw = np.reshape(np.identity(chi*d),(chi,d,chi,d))
  Hrw = np.reshape(np.identity(chi*d),(d,chi,d,chi))
  norm = mf.calc_energy(S,S,L,L,Hl,Hlw,h0,h1,Hrw,Hr,N)
  print('TEMP norm check', norm / (N+2))
##
  R[1,:,:,:] = np.transpose(np.tensordot(np.diag(K[0,:]),R[1,:,:,:],axes=(1,1)),(1,0,2))
  S[1,:,:,:] = np.transpose(np.tensordot(np.diag(L[0,:]),S[1,:,:,:],axes=(1,1)),(1,0,2))
  T[1,:,:,:] = np.transpose(np.tensordot(np.diag(M[0,:]),T[1,:,:,:],axes=(1,1)),(1,0,2))
  R[np.int(N/2),:,:,:] = np.tensordot(SP,S[np.int(N/2),:,:,:],axes=(0,0))
  S[np.int(N/2),:,:,:] = np.tensordot(SO,S[np.int(N/2),:,:,:],axes=(0,0))
##
##
  Corr0 = np.zeros(N+1,dtype=np.complex128)
  Corr0 = mf.calc_corr2(R,T,np.conj(SP.T),N,d,chi)
  Corr1 = np.zeros(N+1,dtype=np.complex128)
  Corr1 = mf.calc_corr2(S,T,np.conj(SO.T),N,d,chi)

#  Mag = np.zeros(N+1,dtype=np.complex128)
#  Mag = calc_corr2(S,S,sz,N,d,chi)

  f_obj0 = open(file_name0,'w')
  f_obj1 = open(file_name1,'w')
#  print(np.real(E_total),np.real(E_total2),file=f_obj0)
#  print(np.real(E_total),np.real(E_total2),file=f_obj1)
  print >> f_obj0, np.real(E_total),np.real(E_total2)
  print >> f_obj1, np.real(E_total),np.real(E_total2)
  for i in range(1,N+1):
#    print('0 ',np.int(i-N/2),np.real(Corr0[i]),np.imag(Corr0[i]),file=f_obj0)
    print >> f_obj0, '0 ',i-N/2,np.real(Corr0[i]),np.imag(Corr0[i])
#    print('0 ',np.int(i-N/2),np.real(Corr1[i]),np.imag(Corr1[i]),file=f_obj1)
    print >> f_obj1, '0 ',i-N/2,np.real(Corr1[i]),np.imag(Corr1[i])
  f_obj0.close()
  f_obj1.close()


norm = 1.0
h0 = np.reshape(np.identity(d*d),(d,d,d,d))
h1 = np.reshape(np.identity(d*d),(d,d,d,d))
Hl = np.reshape(np.identity(chi),(chi,chi))
Hr = np.reshape(np.identity(chi),(chi,chi))
Hlw = np.reshape(np.identity(chi*d),(chi,d,chi,d))
Hrw = np.reshape(np.identity(chi*d),(d,chi,d,chi))
start = time.time()
for t_step in range(t_start,t_max) :
    t = np.imag(i_delta) * np.float(t_step)
    print('TEMP t=',t)
    start = time.time()
    R, K = mf.single_step(R,K,TE,TO,Tl,Tlw,Tr,Trw,chi,chi_max,N,d,norm)
    S, L = mf.single_step(S,L,TE,TO,Tl,Tlw,Tr,Trw,chi,chi_max,N,d,norm)
    T, M = mf.single_step(T,M,TE,TO,Tl,Tlw,Tr,Trw,chi,chi_max,N,d,norm)
##
    print('TEMP cost for a single step :',time.time() - start)
##
    if ( np.mod(t_step,1) == 0 ):
      start = time.time()
      Corr0 = mf.calc_corr2(R,T,np.conj(SP.T),N,d,chi)
      Corr1 = mf.calc_corr2(S,T,np.conj(SO.T),N,d,chi)
#      Mag = calc_corr2(S,S,sz,N,d,chi)
      print('TEMP cost for Corr :',time.time() - start)

      if (np.mod(t_step,1) == 0) :
        f_obj0=open(file_name0,'a')
        f_obj1=open(file_name1,'a')
        for i in range(1,N+1):
#          print(t,np.int(i-N/2),np.real(Corr0[i]),np.imag(Corr0[i]),file=f_obj0)
#          print(t,np.int(i-N/2),np.real(Corr1[i]),np.imag(Corr1[i]),file=f_obj1)
          print >> f_obj0, t,i-N/2,np.real(Corr0[i]),np.imag(Corr0[i])
          print >> f_obj1, t,i-N/2,np.real(Corr1[i]),np.imag(Corr1[i])
        f_obj0.close()
        f_obj1.close()
###
      if ( np.mod(t_step,25) == 0 ):
        t_c = min(t_step+1,t_max)
        mf.pack_parameters(chi,d,chi_max,N,i_delta,t_c,J,alp,Dz,r,g,hz,hst) 
        mf.dump_write(R,K,S,L,T,M,TE,TO,Tl,Tlw,Tr,Trw)
        norm = mf.calc_energy(T,T,M,M,Hl,Hlw,h0,h1,Hrw,Hr,N)
        print('TEMP T_norm check', norm / (N+2))
        del Corr0
        del Corr1
        gc.collect()

###
mf.pack_parameters(chi,d,chi_max,N,i_delta,t_max,J,alp,Dz,r,g,hz,hst) 
mf.dump_write(R,K,S,L,T,M,TE,TO,Tl,Tlw,Tr,Trw)




  
