import abc
import numpy as np

class Loss(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, x, y, w, lambda_):
        pass

    @abc.abstractmethod
    def obj(self, x, y, w, lambda_):
        pass


class LogisticLoss(Loss): # different logistic ridge regression and don't use it anymore

    # def __init__(self, name):
    #     pass
    
    def sigmoid (self, vector ):
        return np.array( list( map( lambda x: 1./(1. + np.exp(-x)), vector ) ) )

    def grad(self, x, y, w, lambda_):
        return (-1*x.T * (y - self.sigmoid(x.dot(w))).T + 2*lambda_*w)*1./x.shape[0]

    def obj(self, x, y, w, lambda_):
        return (-((y.dot( np.log( self.sigmoid(x.dot(w)) ) )) + (( 1- y ).dot( np.log(1- self.sigmoid(x.dot(w))) ))) + (lambda_)*np.square( np.linalg.norm(w)))*1./x.shape[0]


class LogisticLoss_version2(Loss): # logistic ridge regression

    # def __init__(self, name):
    #     pass
    def sigmoid (self, vector ):
        return np.array( list( map( lambda x: 1./(1. + np.exp(-x)), vector ) ) )

    def grad(self, x, y, w, lambda_):
        #return (-1*x.T * (y - self.sigmoid(x.dot(w))).T + 2*lambda_*w)*1./x.shape[0]
        return ( - x.T.dot( np.multiply( self.sigmoid( - np.multiply( y, x.dot( w )) ), y))*1./x.shape[0]+ 2*lambda_*w)
    
    def inv_H_sk( self, x, y, w, lambda_,sk) :
        dot = np.multiply( y, x.dot( w ) )
        tmp1 = np.multiply( self.sigmoid( - dot) , self.sigmoid(  dot ))
        #tmp2 = np.multiply( tmp1, y)
        part1 = x.T.dot(np.diag( tmp1))
        part2 = x.dot( sk )
        return part1.dot( part2 )*1.0/x.shape[0] + 2*lambda_*sk

    def obj(self, x, y, w, lambda_):
        #return (-((y.dot( np.log( self.sigmoid(x.dot(w)) ) )) + (( 1- y ).dot( np.log(1- self.sigmoid(x.dot(w))) ))) + (lambda_)*np.linalg.norm(w))*1./x.shape[0]
        return( np.sum( np.log(1+np.exp( -np.multiply( y, x.dot( w ))))))*1./x.shape[0]+ (lambda_)*np.square(np.linalg.norm(w))
class ridge_regression( Loss ):
    def grad( self, x, y,w, lambda_):
        return( x.T.dot( x.dot( w ) - y) )*2./x.shape[0]+ 2*lambda_*w
    def obj( self, x, y, w, lambda_ ):
        return( np.square( np.linalg.norm( x.dot( w ) - y)))*1./x.shape[0]+ (lambda_)*np.square(np.linalg.norm(w))
    def inv_H_sk( self, x, y, w, lambda_, sk ):
        return( x.T.dot( x.dot(sk ))*1./x.shape[0] + 2*lambda_*sk )
        
class svm_quadratic( Loss ):
    def grad( self, x, y, w, lambda_):
        return  -2*( x.T.dot(np.multiply( y, np.maximum( 0, 1- np.multiply(y , x.dot( w ))))))*1./x.shape[0]+ 2*lambda_*w      
    def obj( self, x, y, w, lambda_ ):
        return( np.square( np.linalg.norm( np.maximum( 0, 1- np.multiply(y , x.dot( w ))))))*1./x.shape[0]+ (lambda_)*np.square(np.linalg.norm(w))
    def inv_H_sk( self, x, y, w, lambda_, sk ):
        flag = np.multiply(y , x.dot( w ))<1.0
        x_new = x[flag, :]
        return( 2*x_new.T.dot( x_new.dot( sk ))/x.shape[0] + 2*lambda_*sk)
        
        
