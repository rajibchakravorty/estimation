# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

    
import sys
sys.path.insert( 0, '../kalmanfilter')
sys.path.insert( 0, '../utility' )

import numpy as np
import numpy.linalg as lin

from numpy import dot

from kalmanfilter.kalman_filter import KalmanFilter
from pdafilter.pda_filter       import PDAFilter
from utility.helperfunc         import guassmix

from numpy import dot

from numpy.linalg import inv


class IPDAFilter( object ):
    
    '''
    Solves the estimation problem with the
    following linear model
    
        X(t+1) = FX(t) + BU(t) + w(k)
        Y(t+1) = HX(t+1) + v(k+1)
     
    where w(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance Q,
          v(k) is drawn randomly from a Gaussian distribution with mean 0 and covariance R
          
    
    Note:
    
        a) The models are linear
        b) The state, measurement, the noises are distributed normally
        c) The sensor can detect the target with a non-unity probability
           , or in other words, the the sensor can miss the target
           This is modeled by PD where 0 < PD <= 1.0
        d) The sensor can pick up measurements that are not originated from the
           target (aka clutter). This implementation assumes a non-parametric 
           temporal distribution and uniform spatial distribution of clutter.
        e) The existence of target is uncertain.
        
        The uncertainty of target existence is solved in following manner:
        
        Successive measurements are used to "initiate" tracks automatically. However, once
        initiated the algorithm lets the track exist in "two states" - "existing" and
        "non-existing". The switching between these two states is modeled as a 22-state
        markov state transition with defined probability (). This has one constraint - once
        a track is "non-existing" it cannot come back to "existing"
           
'''
    
    
    def initModel( self, F, H, Q, R, B, U, probDetection = 0.9, lam = 0, trackStayProbability ):
        
        self.pdaf = PDAFilter( F, H, Q, R, B, U, probDetection = 0.9, lam = 0 )        
        
        self.trackStayProbability = trackStayProbability


    def predict( self, x0, P0, trackExistenceProb ):
        
        [x10, P10] = self.pdaf.predict( x0, P0 )
        
        predTrackTransProbs = self.trackTransitionMatrix * trackStayProbability
        
        return x10, P10, predTrackTransProbs
        
        

    def predictedMeas( self, x10, P10 ):
        
        return self.pdaf.predictedMeas( x10, P10 )
        
        

    def innov( self, yi, yhat ):
        
        return self.pdaf.innov( yi, yhat )
        
    
    def filterUpdate( self, x10, P10, ytilde, S ) :
        
        x11, p11 = self.kf.update( x10, P10, ytilde, S)
        beta     = np.exp( -0.5 * dot( ytilde.T, dot( lin.inv( S ), ytilde  ) ) )
        return x11, p11, beta
    
  
    
    
    def runFilter( self, x0, p0, ys, gateProbability, gateVolume ):
        
        
                
        x10, p10 = self.predict( x0, p0 )
        
        
        
        
        yhat, S = self.predictedMeas( x10, p10 )
        
        if( ys == None ):
            
            totalReturns = 0 
        else:
            totalReturns = ys.shape[1]
        
        stateSize = self.kf.F.shape[0]
        measSize  = self.kf.H.shape[0]
        
        x11s = np.zeros( (stateSize, totalReturns + 1 ) )
        p11s = np.zeros( ( stateSize, stateSize, totalReturns+ 1 ) )
        betas = np.zeros( totalReturns + 1 )
        
        x11s[ :, 0 ]    = np.reshape( x10, (stateSize ) )
        p11s[ :, :, 0 ] = np.reshape( p10, ( stateSize, stateSize ) )
        
        betas[ 0 ] = ( 1-self.PD*gateProbability) * gateProbability * np.sqrt( lin.det( 2.0 * np.pi * S  ) ) / gateVolume

        
        for m in np.arange( 0, totalReturns ):
            
            Yt = np.reshape( ys[:,m], ( measSize, 1 ) )
            
            ytilde = self.innov( Yt, yhat )

            
            x11, p11, beta = self.filterUpdate( x10, p10, ytilde, S )
            
            x11s[ :, m+1 ]   = np.reshape( x11, ( stateSize, ) )
            p11s[:, :, m+1 ] = np.reshape( p11, (stateSize , stateSize ) )
            
            betas[ m+1 ] = self.PD*gateProbability*beta/totalReturns
            
            
        return x11s, p11s, betas
        
    def mixOutput( self, x11s, p11s, betas ):
        
        if( np.sum( betas ) != 1.0 ):
            
            betas = betas / np.sum( betas )
            
            
        x11, p11 = guassmix( x11s, p11s, betas )
        
        return x11, p11