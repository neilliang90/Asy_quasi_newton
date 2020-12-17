#import sklearn
import sys
import numpy as np
#import pandas as pd
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
import logging
import sklearn

MULTI_L = 1.3
#FILENAME = ''


def slave_loop(loss, lock, L, x_data, x_indices, x_indptr, x_shape, y, batch_size4SVRG, w, w_multi, u,  K, flag, proc_id,  stepsize_type, stepsize,regularizer):
  #proc_id = int(multiprocessing.current_process().name[-1])-1
  print("slave " + str( proc_id) + " started")
  x = csr_matrix((x_data, x_indices, x_indptr), shape=x_shape, copy=False)
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
      
    tttt1 = time.time() 
    for t in range(L):
      #w_p = np.copy(w)
      #print("slave iteration ", k, t)
      #lock.acquire()
      w_cp = w
      sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
      sample_id = sample_id[0]
      g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_cp, regularizer)
      #print(multiprocessing.current_process(), "Grad w ", w_cp[ 0:3], w_cp[ -3:-1])
      #lock.release()
      
      #Hv = twoloop(v, sk, yk)
      Hv= -g1
      lock.acquire()
      #print(multiprocessing.current_process(), "Before Update w ", w[0:3], w[ -3:-1])
      w[:] = w + eta*Hv
      #print(multiprocessing.current_process(), "After Update w ", w[0:3], w[ -3:-1])
      lock.release()
      
      #wp_list[proc_id] = wp_list_multi_L
    #tttt2 = time.time()  
    #print( 'slve time ', tttt2 - tttt1)
    flag[proc_id] = 1
  while flag[0] != 1:
    if flag[ 0 ] == 3:
      break
    pass
  flag[proc_id] = 2
  print("slave " + str( proc_id) + " finished")



def master_loop(loss, lock, L, P,x_data, x_indices, x_indptr, x_shape, y, batch_size4SVRG, w, w_multi, u, K, flag, optgap, stepsize_type, stepsize, regularizer):
  #proc_id = int(multiprocessing.current_process().name[-1])-1
  print("master started")
  x = csr_matrix((x_data, x_indices, x_indptr), shape=x_shape, copy=False)
  flag[0] = 1
 
  #store information
  obj_list  = list();
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
      
    for t in range(L):
      #print( t )
      #w_p = np.copy(w)
      #print("master iteration ", k, t)
      w_cp = w
      #lock.acquire()
      sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
      sample_id = sample_id[0]
      g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_cp, regularizer)
      #print(multiprocessing.current_process(), "Grad w ", w_cp[ 0:3], w_cp[ -3:-1])
      #lock.release()
      
        
        
        #print( 'w_multi : ', np.square( np.linalg.norm( w_multi )))
      
      Hv= -g1
      #print( 'w : ', np.square( np.linalg.norm( w )))
      
      #print( np.square( np.linalg.norm( Hv )))
      lock.acquire()
      #print(multiprocessing.current_process(), "Before Update w ", w[0:3], w[ -3:-1])
      w[:] = w + eta*Hv
      #print(multiprocessing.current_process(), "After Update w ", w[0:3], w[ -3:-1])
      lock.release()

     
    #--------------------------------------
    while any(flag[1:] != 1):
      pass
    

    #print(  "objective value  %.30f" % loss.obj(x,y,w,1./x.shape[0]))
    
    # update w and compute w_multi
      #print( x.shape[0]//(batch_size4SVRG*L*P))
    #w_multi[:] = np.copy(w)
    #u[:] = loss.grad(x, y, w, regularizer)
    obj_temp = loss.obj( x, y, w, regularizer )
    #if k>0:
    time_k = time.time()
    print( 'Epoch: '+ str(k+1) +', data passes : '+ str((k+1))+ ', time :' + str(time_k - t0 ))
        #store informations
        #if verbose:
    print(  "objective value  %.30f" % obj_temp)
    #print( "Norm for w : ", np.square( np.linalg.norm( w )))
    obj_list.append(obj_temp)
    if k>0 and np.abs( obj_list[-1] - obj_list[ -2 ] ) < optgap:
      print( "Optimality gap tolerance reached : optgap " + str( optgap ))
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


def ParallelSQN( f, epoch, P, batch_size4SVRG , stepsize= 10**-5,  stepsize_type = "fixed", verbose = False, optgap = 10**(-30 ), loss = 'logistic'):
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
  # global FILENAME
  # FILENAME = outfile

  # print FILENAME
  # logging.basicConfig(level=logging.DEBUG, filename=FILENAME, filemode='w')

  x, label = Util.readlibsvm(f)
  #x = sklearn.preprocessing.normalize( x )
  #label = (label == 1 )* 1
  
  L = x.shape[0]//( batch_size4SVRG*P)
  
  if loss=='logistic':
    loss = Loss.LogisticLoss_version2()
  elif loss == 'svm' :
    loss = Loss.svm_quadratic()
  elif loss == 'ridge' :
    loss = Loss.ridge_regression()
  #regularizer = 1/x.shape[0]
  regularizer = 10**(-3)
  #loss = Loss.ridge_regression()
  print( "The number of cores : " + str( multiprocessing.cpu_count() ))
  print( "The dataset : " + str( f) )
  print( "The number of instances N : " + str( x.shape[0]) )
  print( "The number of features p : " + str( x.shape[1]) )
  print( 'The number of processes P : '+ str( P ) )
  print( "The batch size for SVRG : " + str( batch_size4SVRG ) )
  print( "The step size : " + str(  stepsize ) )
  print( "The epoch : " + str(  epoch ) )
  print( "The loss type: " + str(loss) )
  print( "The regularizer : " + str(  regularizer ) )
  
  
  # init shared mem variables
  x_data = sharedmem.empty( len( x.data ), dtype=x.data.dtype)
  x_data = x.data
  x_indices = sharedmem.empty( len( x.indices), dtype = x.indices.dtype )
  x_indices = x.indices
  x_indptr = sharedmem.empty( len( x.indptr), dtype = x.indptr.dtype)
  x_indptr = x.indptr
  y = sharedmem.empty( len( label), dtype = label.dtype)
  y = label
  x_shape = sharedmem.empty( len( x.shape), dtype = x.indices.dtype)
  x_shape = x.shape

     
  lock = Lock()
  w = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  #w[:] = np.random.rand(x.shape[1],)#Array(c_double, np.random.rand(x.shape[0]), lock=False)
  w[:] = np.zeros(x.shape[1],)
  w_multi = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  w_multi[:] = np.copy(w) #multiprocessing.sharedctypes.copy(w)
  u = sharedmem.empty(x.shape[1],dtype = np.longdouble)
  u[:] = np.random.rand(x.shape[1],)

  flag = sharedmem.empty(P, dtype=int)
  flag[:] = np.zeros([P,])
 
  # ----------------------------------------------

  procs = []

  # add master
  procs.append(Process(target=master_loop, args=(loss, lock, L, P, x_data, x_indices, x_indptr,x_shape,  y, batch_size4SVRG, w, w_multi, u,  epoch,  flag, optgap, stepsize_type, stepsize, regularizer )))
  # add slaves
  for proc_id in range(1, P):
    t = Process(target=slave_loop, args=(loss, lock, L, x_data, x_indices, x_indptr,x_shape,  y, batch_size4SVRG, w, w_multi, u,  epoch, flag, proc_id, stepsize_type, stepsize, regularizer ))
    procs.append(t)

  # start all processes
  for t in procs:
    t.daemon = True
    t.start()
  # wait until all processes finish
  for t in procs:
    t.join()
    
    
  print( 'Finish parallel ')
  # t1 = time.time()
  # print( "Time : ", t1-t0)
  # return [obj_list, time_list , datapasses_list]

def main():
  #f = sys.argv[1]
  #dataset = 'real-sim'
  f = '/Users/Neil/Desktop/rcv1_train.binary'
  #f = '/Users/Neil/Desktop/E2006.train'
  #f = '../data/' + dataset
  #f= '../data/real-sim'
  batch_size4SVRG = 100
  #eta = float(sys.argv[2]) #20
  eta=0.1
  #memory = int(sys.argv[2])
  #epoch = int(sys.argv[3])
  #P = int(sys.argv[4])
  #L = int(sys.argv[5])
  epoch = 100
  P= 4
  loss = 'svm'
  #f_name = 'svrg'+dataset + '_thr_' + str( P ) +'_b4SVRG_' + str( batch_size4SVRG )  + '_step_'+str( eta ) + loss
  #output_file = '../log/guannan/'+f_name
  #logging.basicConfig(filename=output_file, level=logging.DEBUG)
  logging.basicConfig( level=logging.DEBUG)
  
  ParallelSQN( f, epoch, P, batch_size4SVRG,  eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss )
  
  # for eta in etas:
    # f_name = 'svrg'+dataset + '_thr_' + str( P ) +'_b4SVRG_' + str( batch_size4SVRG )  + '_step_'+str( eta ) + loss
	# output_file = '../log/guannan/'+f_name
    # logger = logging.getLogger()
    # logger.handlers=[]
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(output_file)
    # logger.addHandler(handler)
    # ParallelSQN( f, epoch, P, batch_size4SVRG,  eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss )
  

if __name__== "__main__":
  main()
