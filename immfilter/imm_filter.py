# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

from numpy import dot

from numpy.linalg import inv


class IMMFilter( object ):
    
    '''
    Solves the estimation problem with the
    following linear model
    
        X(t+1) = FX(t) + BU(t) + w(k)
        Y(t+1) = HX(t+1) + v(k+1)
     
    where w(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance Q,
          v(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance R
          U(t) is one of the model inputs.
          
    The target can switch models. The filter models this switch as a Markov State Transition
    T = { pi_{ij}; i, j <= M } where M is the number of models and
    pi_{ij} = P( M_k = j | M_(k-1) = i ), M_k = j denotes that the model at time k was j.
    
    

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
        yhat, S  = self.predictedMeas( x10, P10 )
        ytilde   = self.innov( y1, yhat)
        x11, p11 = self.update( x10, P10, ytilde, S )
        
        return x11, p11
    
    def predict( self, x0, P0 ):
        
        x10 = dot( self.F, x0 ) + self.BU
        
        P10 = dot( self.F, dot( P0, self.F.T ) ) + self.Q
        
        return x10,P10
        
    def predictedMeas( self, x10, P10 ):
        
        yhat = dot( self.H, x10 )
        
        S = dot( self.H, dot( P10, self.H.T ) ) + self.R
        
        return yhat, S        
        
    def innov( self, y1, yhat ):
        
        ytilde = y1 - yhat

        return ytilde
    
    def update( self, x10, P10, ytilde, S ):
        
        K   = dot( P10, dot( self.H.T, inv( S ) ) )
        x11 = x10 + dot( K, ytilde )
        P11 = P10 - dot( K, dot( self.H, P10 ) )

        return x11, P11

