# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

import sys
sys.path.insert( 0, '../kalmanfilter')

from kalmanfilter.kalman_filter import KalmanFilter


from numpy import dot
from numpy.linalg import inv


class PDAFilter( object ):
    
    
    def initModel( self, F, H, Q, R, B, U, probDetection = 0.9, lam = 0 ):
        
        self.kf = KalmanFilter()
    
        self.kf.initModel( F, H, Q, R, B, U )
    
        self.PD = probDetection
        
        self.lam = lam
    
    def predict( self, x0, P0 ):
        
        return self.kf.predict( x0, P0 )
        
        
    def predictedMeas( self, x10, P10 ):
        
        return self.kf.predictedMeas( x10, P10 )
        
        
    def innov( self, yi, yhat ):
        
        return self.kf.innov( yi, yhat )