# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

from numpy import dot, sum, tile, linalg
from numpy.linalg import inv


class KalmanFilter( object ):
    
    '''
    Solves the estimation problem with the
    following linear model
    
        X(t+1) = FX(t) + BU(t) + w(k)
        Y(t+1) = HX(t+1) + v(k+1)
     
    where w(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance Q
    and   v(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance R


    Note:

        a) The models are linear
        b) The state, measurement, the noises are distributed normally
        c) 
    '''    
    
    
    
    def initModel( self, F, H, Q, R, B, U ):
        
        self.F = F;
        
        self.H = H;
        
        self.Q = Q;
        
        self.R = R;
        
        self.BU = dot( B, U );
        
    
    def runFilter( self, x0, P0, y1 ):
        
        x10, P10 = self.predict( x0, P0 )
        yhat, S  = self.innov( y1, x10, P10 )
        x11, p11 = self.update( x10, P10, yhat, S )
        
        return x11, p11
    
    def predict( self, x0, P0 ):
        
        x10 = dot( self.F, x0 ) + self.BU
        
        P10 = dot( self.F, dot( P0, self.F.T ) ) + self.Q
        
        return x10,P10
        
    def innov( self, y1, x10, P10 ):
        
        ytilde = y1 - dot( self.H, x10 )
        
        S = dot( self.H, dot( P10, self.H.T ) ) + self.R
        
        return ytilde, S
    
    def update( self, x10, P10, ytilde, S ):
        
        K   = dot( P10, dot( self.H.T, inv( S ) ) )
        x11 = x10 + dot( K, ytilde )
        P11 = P10 - dot( K, dot( self.H, P10 ) )

        return x11, P11