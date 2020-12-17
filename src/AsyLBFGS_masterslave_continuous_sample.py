import numpy as np

import multiprocessing
from multiprocessing import Process, Lock, current_process


import sharedmem
from functools import reduce
from scipy.sparse import csr_matrix

import time
import Loss
import Util
import logging
import logging.config

def twoloop(grad, sk, yk, k):
  #logging.debug( "start two loop recursion")
  #logging.debug( sk )
  rk = np.zeros(sk.shape[1],)
  a = np.zeros(sk.shape[1],)
  
  if k == 0:
    Hv = -np.copy(grad)

  else:
    if k <= 10:
        pos_range = list( range(k) )
        neg_range = list( range(k-1, -1, -1) )
    else:
        neg_range = list( range(k%10-1-1, -1, -1)) + list( range(9, k%10-1-1, -1))
        pos_range = reversed(neg_range)
    
    q = np.copy(grad)
    for i in neg_range:
      rk[i] = 1/yk[:,i].dot(sk[:,i])
      a[i] = rk[i]*sk[:,i].dot(q)
      q = q - a[i]*yk[:,i]

    Hk0 = sk[:,neg_range[0]].dot(yk[:,neg_range[0]]) / (yk[:,neg_range[0]].dot(yk[:,neg_range[0]]))
    R = Hk0 * q

    for j in pos_range:
      beta = rk[j]*(yk[:,j].dot(R))
      R = R + sk[:,j]*(a[j]-beta)
    Hv = -R
  return Hv



def slave_loop(loss, lock, L, x_data, x_indices, x_indptr, x_shape, y, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, proc_id,  stepsize_type, stepsize, regularizer):
  global FILENAME

  logging.debug("slave "+ str(proc_id)+ " started")
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
    #logging.debug("slave ", proc_id, " " , k)
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
      #logging.debug("slave iteration ", k, t)
      if k <1:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #logging.debug(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        v = g1
      else:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #logging.debug(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        g2 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_multi, regularizer )
        v = g1 - g2 + u 
      
      
      Hv = twoloop(v, sk, yk, k)
      #Hv= v
      lock.acquire()
      w[:] = w + eta*Hv
      wp_list[proc_id]=  wp_list[proc_id] + w
      lock.release()
      
      #wp_list[proc_id] = wp_list_multi_L
    flag[proc_id] = 1
    wp_list[proc_id].fill(0)
  flag[proc_id] = 2
  logging.debug("slave "+str( proc_id)+ " finished")



def master_loop(loss, lock, L, P, memory, x_data, x_indices, x_indptr, x_shape, y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, optgap, stepsize_type, stepsize, regularizer):
 #proc_id = int(multiprocessing.current_process().name[-1])-1
  logging.debug("master started")
  x = csr_matrix((x_data, x_indices, x_indptr), shape=x_shape, copy=False)
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
    #logging.debug("master iteration ", k)
      #setup stepsize
    if stepsize_type == "fixed":
      eta = stepsize
    elif stepsize_type == "decay":
      eta = 1./( k+1) 
    elif stepsize_type == "sqrtdecay":
      eta = stepsize*1./np.sqrt( k +1)
    elif stepsize_type == "squaredecay":
      eta = ( 1/np.square(k + 1 ))
      
      
    #logging.debug( k )
    wp_list[0].fill(0)
    for t in range(L):
      #w_p = np.copy(w)
      #logging.debug("master iteration ", k, t)
      if k <1:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #logging.debug(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        v = g1
      else:
        lock.acquire()
        sample_id = np.random.randint(x.shape[0]-batch_size4SVRG, size=1)
        sample_id = sample_id[0]
        g1 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w, regularizer)
        #logging.debug(multiprocessing.current_process(), "Grad w ", w[ 0:3])
        lock.release()
        g2 = loss.grad(x[sample_id:sample_id+batch_size4SVRG], y[sample_id:sample_id+batch_size4SVRG], w_multi, regularizer)
        v = g1 - g2 + u 
      
      Hv = twoloop(v, sk, yk, k)
      #logging.debug( 'Hv : ', np.square( np.linalg.norm( Hv )))
      
      lock.acquire()
      w[:] = w + eta*Hv
      wp_list[0] = wp_list[0] + w
      lock.release()


    #--------------------------------------
    while any(flag[1:] != 1):
      if all(flag[ 1: ] == 2 ):
        break
      pass

    
    sample_id = np.random.randint(x.shape[0]-batch_size4SVRG*10, size=1)
    sample_id = sample_id[0]
    w_new = reduce(lambda a,b: a + b, wp_list)
    w_new = w_new/(P*L)
    tmp = w_new - w_old
    tmp1 = loss.inv_H_sk(x[sample_id:sample_id+10*batch_size4SVRG], y[sample_id:sample_id+10*batch_size4SVRG], w_new, regularizer, tmp)
    
    
    if k < memory:
      sk[:,k] = tmp
      yk[:,k] = tmp1
    else:
      sk[:,k%10] = tmp
      yk[:,k%10] = tmp1
    w_old = np.copy(w_new)

    #logging.debug(  "objective value  %.30f" % loss.obj(x,y,w,1./x.shape[0]))
    
    # update w and compute w_multi
    if k %(x.shape[0]//(batch_size4SVRG*L*P )) == 0:
      #logging.debug( x.shape[0]//(batch_size4SVRG*L*P))
      w_multi[:] = np.copy(w)
      u[:] = loss.grad(x, y, w,regularizer)
      if k>1:
        epoch_count =  k //(x.shape[0]//(batch_size4SVRG*L*P ))
        datapass_count = epoch_count*( 2 + 1*( batch_size4H/(batch_size4SVRG*L*P ) ))
        time_k = time.time()
        logging.debug( 'Epoch: '+str(epoch_count) + ', data passes : '+str(datapass_count)+ ', time :' + str(time_k - t0 ))
        #store informations
        
        obj_temp = loss.obj( x, y, w, regularizer )
        #if verbose:
        logging.debug(  "objective value  %.30f" % obj_temp)
        obj_list.append(obj_temp)
        datapasses_list.append( datapass_count )
        time_list.append( time_k - t0 )
        if epoch_count>2 and np.abs( obj_list[-1] - obj_list[ -2 ] ) < optgap:
          logging.debug( "Optimality gap tolerance reached : optgap "+str( optgap ))
          flag[0]=3
          break
         
    flag[0] = 1
    while any(flag[1:] != 2):
      pass
    #logging.debug(flag)
    flag[0] = 0
    
  flag[0]=3
  logging.debug("master finish")
  #return obj_list, datapasses_list, time_list
  #exit() 


def ParallelSQN( f, epoch, P, L, memory, batch_size4SVRG , batch_size4H , stepsize= 10**-5,  stepsize_type = "fixed", verbose = False, optgap = 10**(-30 ), loss = 'logistic'):
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
  #label = (label == 1 )* 1
  #x = sklearn.preprocessing.normalize( x )
  
  if loss=='logistic':
    loss = Loss.LogisticLoss_version2()
  elif loss == 'svm' :
    loss = Loss.svm_quadratic()
  elif loss == 'ridge' :
    loss = Loss.ridge_regression()
  regularizer = 10**(-6)
  #regularizer = 1./x.shape[0]
  logging.debug( regularizer )
  logging.debug( "--->>>>>>--------------------------->>>>>>>>" )
  
  logging.debug( "The number of cores : "+str( multiprocessing.cpu_count() ))
  logging.debug( "The dataset : "+str(f ))
  logging.debug( "The number of instances N : "+ str(x.shape[0]))
  logging.debug( "The number of features p : "+ str(x.shape[1]))
  logging.debug( 'The number of processes P : '+ str(P ))
  logging.debug( "The batch size for SVRG : "+ str(batch_size4SVRG ))
  logging.debug( "The batch size for Hassion of SQN : "+ str( batch_size4H ))
  logging.debug( "The L in SQN : "+ str(L ))
  logging.debug( "The step size : "+ str(stepsize ))
  logging.debug( "The epoch : "+ str( epoch ))
  logging.debug( "The regularizer : "+str(regularizer ))
  
  
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

  K = epoch*(x.shape[0]//(batch_size4SVRG*L*P ) )
  wp_list = sharedmem.empty([P, x.shape[1]])
  wp_list[:] = np.zeros([P, x.shape[1]]) 
  procs = []
 
  
  #master_loop(loss, lock, L, P, memory, x_data, x_indices, x_indptr,x_shape,  y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K,  flag, optgap, stepsize_type, stepsize, regularizer)
  # add master
  procs.append(Process(target=master_loop, args=(loss, lock, L, P, memory, x_data, x_indices, x_indptr,x_shape,  y, batch_size4H, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K,  flag, optgap, stepsize_type, stepsize, regularizer)))
  # add slaves
  for proc_id in range(1, P):
    t = Process(target=slave_loop, args=(loss, lock, L, x_data, x_indices, x_indptr,x_shape,  y, batch_size4SVRG, w, w_multi, u, wp_list, sk, yk, K, flag, proc_id, stepsize_type, stepsize, regularizer))
    procs.append(t)

  # start all processes
  for t in procs:
    t.daemon = True
    t.start()
  # wait until all processes finish
  for t in procs:
    t.join()
    
    
  logging.debug( 'Finish parallel ')
  # t1 = time.time()
  # logging.debug( "Time : ", t1-t0)
  # return [obj_list, time_list , datapasses_list]

def main():
  #f = sys.argv[1]
  #dataset = 'news20.binary'
  #f = '../data/' + dataset
  #f = '/Users/Neil/Desktop/mnist'
  #f = '/Users/Neil/Desktop/rcv1_test.binary'
  #f = '../data/real-sim'
  #f = '/Users/Neil/Dropbox/Parallel SQN/rcv1.bin'
  f = '/Users/Neil/Desktop/rcv1_train.binary'
  #f = '/Users/Neil/Desktop/E2006.train'
  batch_size4SVRG = 10
  batch_size4H = 10*batch_size4SVRG
  #eta = float(sys.argv[2]) #20
  eta=0.015
  #memory = int(sys.argv[2])
  memory = 10
  #epoch = int(sys.argv[3])
  #P = int(sys.argv[4])
  #L = int(sys.argv[5])
  epoch = 100
  P= 1
  L = 10
  loss = 'svm'
  #f_name = dataset + '_thr_' + str( P ) + '_L_' + str( L ) +'_M_' + str( memory) +'_b4SVRG_' + str( batch_size4SVRG ) + '_b4H_' + str( batch_size4H) + '_step_'+str( eta ) + loss
  #output_file = '../log/guannan/'+f_name
  #logging.basicConfig(filename=output_file, level=logging.DEBUG)
  logging.basicConfig(level=logging.DEBUG)
  
  ParallelSQN(f, epoch, P, L, memory, batch_size4SVRG , batch_size4H,  eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss)
  

if __name__== "__main__":
  main()
