#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 19:04:24 2018

@author: Neil
"""

#import sklearn
import sys
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Process, Lock, current_process
#import copy
#import time
#from operator import add
import sharedmem
from functools import reduce
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import time
import Loss
import Util

MULTI_L = 1.3

def twoloop(grad, sk, yk, k, step_epoch):
  #print( "start two loop recursion")
  #print( sk )
  rk = np.zeros(sk.shape[1],)
  a = np.zeros(sk.shape[1],)
  
  if k < step_epoch:
    Hv = -np.copy(grad)

  else:
    if k <= 10:
        pos_range = list( range(k) )
        neg_range = list( range(k-1, -1, -1) )
    else:
        neg_range = list( range(k%10-1-1, -1, -1)) + list( range(9, k%10-1-1, -1))
        pos_range = reversed(neg_range)
    
    #print( neg_range )
    #print( )
    #print( '********************************#')
    q = np.copy(grad)
    
    for i in neg_range:
      rk[i] = 1/yk[:,i].dot(sk[:,i])
      a[i] = rk[i]*sk[:,i].dot(q)
      q = q - a[i]*yk[:,i]

    Hk0 = sk[:,neg_range[0]].dot(yk[:,neg_range[0]]) / (yk[:,neg_range[0]].dot(yk[:,neg_range[0]]))
    R = Hk0 * q
    #print( 'R : ', np.square( np.linalg.norm( R )))
    for j in pos_range:
      beta = rk[j]*(yk[:,j].dot(R))
      R = R + sk[:,j]*(a[j]-beta)
    Hv = -R
  return Hv



def slave_loop(loss, lock, L, x_data, y, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, proc_id,  stepsize_type, stepsize, regularizer, step_epoch):
  #proc_id = int(multiprocessing.current_process().name[-1])-1
  print("slave ", proc_id, " started")
  #x = csr_matrix((x_data, x_indices, x_indptr), shape=x_shape, copy=False)
  x = x_data
  #wp_list_multi_L = np.copy(wp_list[proc_id])
  #global MULTI_L
  
  for k in range (K):
    if flag[ 0 ] == 3:
      break
    while flag[0] != 1:
      if flag[ 0 ] == 3:
        break
      pass
    flag[proc_id] = 2
    #print("slave iteration ", k)
    #setup stepsize
    if stepsize_type == "fixed":
      eta = stepsize
    elif stepsize_type == "decay":
      eta = 1./( k+1) 
    elif stepsize_type == "sqrtdecay":
    #eta = 1/np.sqrt(k*L + t +1 )
      eta = stepsize*1./np.sqrt( k +1)
    elif stepsize_type == "squaredecay":
      eta = ( 1/np.square(k + 1 ))
      

    for t in range(L):
      #w_p = np.copy(w)
      #print("slave iteration ", k, t)
      if k <1:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        v = g1
      else:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        g2 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_multi, regularizer )
        v = g1 - g2 + u 
      
      Hv = twoloop(v, sk, yk, k, step_epoch)
      #Hv= -v
      lock.acquire()
      w[:] = w + eta*Hv
      wp_list[proc_id]=  wp_list[proc_id] + w
      lock.release()
      
      #wp_list[proc_id] = wp_list_multi_L
    while flag[0] != 0:
        #print( flag )
        if flag[ 0 ] == 3:
          break
        pass
    flag[proc_id] = 1
    wp_list[proc_id].fill(0)
  while flag[0] != 1:
    if flag[ 0 ] == 3:
      break
    pass
  flag[proc_id] = 2
  print("slave ", proc_id, " finished")



def master_loop(loss, lock, L, P, memory, x_data, y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, optgap, stepsize_type, stepsize, regularizer, step_epoch):
  #proc_id = int(multiprocessing.current_process().name[-1])-1
  print("master started")
  #x = csr_matrix((x_data, x_indices, x_indptr), shape=x_shape, copy=False)
  x = x_data
  w_old = np.copy(w)
  flag[0] = 1 
  
 
  #store information
  obj_list  = list();
  time_list = list();
  datapasses_list = list( )
  t0 = time.time()
  while any(flag[1:] != 2):
        pass
  flag[0] = 0
  for k in range (K):
    #print("master iteration ", k)
      #setup stepsize
    if stepsize_type == "fixed":
      eta = stepsize
    elif stepsize_type == "decay":
      eta = 1./( k+1) 
    elif stepsize_type == "sqrtdecay":
    #eta = 1/np.sqrt(k*L + t +1 )
      eta = stepsize*1./np.sqrt( k +1)
    elif stepsize_type == "squaredecay":
      eta = ( 1/np.square(k + 1 ))
      
      
    #print( k )
    wp_list[0].fill(0)
    for t in range(L):
      #w_p = np.copy(w)
      #print("master iteration ", k, t)
      if k <1:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        v = g1
      else:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        g2 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_multi, regularizer)
        v = g1 - g2 + u 
      
      Hv = twoloop(v, sk, yk, k, step_epoch)
      #print( 'Hv : ', np.square( np.linalg.norm( Hv )))
      #Hv= -v
      lock.acquire()
      w[:] = w + eta*Hv
      wp_list[0] = wp_list[0] + w
      lock.release()


    #--------------------------------------
    while any(flag[1:] != 1):
      pass

    #print("master in loop ", k)
    sample_id = np.random.randint(x.shape[0]-batch_size4SVRG*10, size=1)
    sample_id = sample_id[0]
    w_new = reduce(lambda a,b: a + b, wp_list)
    w_new = w_new/(P*L)
    tmp = w_new - w_old
    #tmp = tmp.reshape(tmp.shape[0], 1)
    #tmp1 = loss.grad(x[sample_id:sample_id+10*batch_size4SVRG], y[sample_id:sample_id+10*batch_size4SVRG], w_new, regularizer) - loss.grad(x[sample_id:sample_id+batch_size4SVRG*10], y[sample_id:sample_id+batch_size4SVRG*10], w_old, regularizer)
    tmp1 = loss.inv_H_sk(x[sample_id:sample_id+10*batch_size4SVRG], y[sample_id:sample_id+10*batch_size4SVRG], w_new, regularizer, tmp)
    #tmp1 = tmp1.reshape(tmp1.shape[0], 1)
    #print( tmp )
    

    if k < memory:
      sk[:,k] = tmp
      yk[:,k] = tmp1
    else:
      sk[:,k%10] = tmp
      yk[:,k%10] = tmp1
    w_old = np.copy(w_new)

    #print(  "objective value  %.30f" % loss.obj(x,y,w,1./x.shape[0]))
    
    # update w and compute w_multi
    if k %(x.shape[0]//(batch_size4SVRG*L*P )) == 0:
      #print( x.shape[0]//(batch_size4SVRG*L*P))
      w_multi[:] = np.copy(w)
      u[:] = loss.grad(x, y, w,regularizer)
      if k>1:
        epoch_count =  k //(x.shape[0]//(batch_size4SVRG*L*P ))
        datapass_count = epoch_count*( 2 + 2*( batch_size4H/(batch_size4SVRG*L*P ) ))
        time_k = time.time()
        print( 'Epoch: ', epoch_count , ', data passes : ',datapass_count, ', time :' , time_k - t0 )
        #store informations
        
        obj_temp = loss.obj( x, y, w, regularizer )
        #if verbose:
        print(  "objective value  %.30f" % obj_temp)
        obj_list.append(obj_temp)
        datapasses_list.append( datapass_count )
        time_list.append( time_k - t0 )
        if epoch_count>1 and np.abs( obj_list[-1] - obj_list[ -2 ] ) < optgap:
          print( "Optimality gap tolerance reached : optgap ", optgap )
          flag[0]=3
          break
          #return obj_list, datapasses_list, time_list
         

    flag[0] = 1
    while any(flag[1:] != 2):
      pass
    #print(flag)
    flag[0] = 0
    
  flag[0]=3
  print("master finish")
  #return obj_list, datapasses_list, time_list
  #exit() 


def ParallelSQN( x, label, epoch, P, L, memory, batch_size4SVRG , batch_size4H , stepsize= 10**-5,  stepsize_type = "fixed", verbose = False, optgap = 10**(-30 )):
  '''
  INPUT:
  x : data
  y : vector for label 1 or 0
  K : number of outest iterations
  P : number of parallel precossors
  batch_size4SVRG :
  batch_size4H
  stepsize  : default 10**-5
  stepsize_type : fixed, decay 1/t, sqrt decay 1/sqrt( t )
  OUTPUT:
  '''
  
  #x, label = Util.readlibsvm(f)
  #x= np.genfromtxt('/Users/Neil/Dropbox/Parallel SQN/experiments/simulations/x_train.csv',delimiter=',')
  #label = np.genfromtxt('/Users/Neil/Dropbox/Parallel SQN/experiments/simulations/y_train.csv',delimiter=',')
  #x= pd.read_csv('/Users/Neil/Dropbox/Parallel SQN/experiments/simulations/x_train.csv')
  #label = pd.read_csv('/Users/Neil/Dropbox/Parallel SQN/experiments/simulations/y_train.csv')
  loss = Loss.ridge_regression()
  #loss = Loss.svm_quadratic()
  regularizer = 0
  print( "The number of cores : ", multiprocessing.cpu_count() )
  #eprint( "The dataset : ", f )
  print( "The number of instances N : ", x.shape[0])
  print( "The number of features p : ", x.shape[1])
  print( 'The number of processes P : ', P )
  print( "The batch size for SVRG : ", batch_size4SVRG )
  print( "The batch size for Hassion of SQN : ", batch_size4H )
  print( "The L in SQN : ", L )
  print( "The step size : ", stepsize )
  print( "The epoch : ", epoch )
  
  
  x_data = sharedmem.empty( x.shape, dtype = x.dtype )
  x_data[:] = x
  y = sharedmem.empty( len( label), dtype = label.dtype)
  y[:] = label

     
  lock = Lock()
  w = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  w[:] = np.random.rand(x.shape[1],)#Array(c_double, np.random.rand(x.shape[0]), lock=False)
  #w[:] = np.zeros(x.shape[1],)
  w_multi = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  w_multi[:] = np.copy(w) #multiprocessing.sharedctypes.copy(w)
  sk = sharedmem.empty([x.shape[1],10], dtype = np.longdouble)
  #sk[,0] = np.ndarray([x.shape[1],0])
  yk = sharedmem.empty([x.shape[1],10], dtype = np.longdouble)
  #yk[:] = np.ndarray([x.shape[1],0])
  u = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  u[:] = np.random.rand(x.shape[1],)

  flag = sharedmem.empty(P, dtype=int)
  flag[:] = np.zeros([P,])
 
  # ----------------------------------------------
  step_epoch = x.shape[0]//(batch_size4SVRG*L*P)
  K = epoch*(x.shape[0]//(batch_size4SVRG*L*P ) )
  wp_list = sharedmem.empty([P, x.shape[1]])
  wp_list[:] = np.zeros([P, x.shape[1]]) 
  procs = []
  
  #master_loop(loss, lock, L, P, memory,  x_data, y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K,  flag, optgap, stepsize_type, stepsize, regularizer)
  # add master
  procs.append(Process(target=master_loop, args=(loss, lock, L, P, memory, x_data, y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K,  flag, optgap, stepsize_type, stepsize, regularizer, step_epoch)))
  # add slaves
  for proc_id in range(1, P):
    t = Process(target=slave_loop, args=(loss, lock, L, x_data , y, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, proc_id, stepsize_type, stepsize, regularizer, step_epoch))
    procs.append(t)

  # start all processes
  for t in procs:
    t.start()
  # wait until all processes finish
  for t in procs:
    t.join()
    
    
  print( 'Finish parallel ')
  # t1 = time.time()
  # print( "Time : ", t1-t0)
  # return [obj_list, time_list , datapasses_list]

def main():
  batch_size4SVRG = 10
  batch_size4H = 100
  eta=0.02
  memory = 10
  epoch = 100
  P= 1
  L = 80
  
  AsyLBFGS_dense_version_V2.ParallelSQN(A, y,  epoch, P, L, memory, batch_size4SVRG , batch_size4H,  eta)
  

if __name__== "__main__":
  main()