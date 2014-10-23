# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:39:51 2014

@author: rchakrav
"""

import numpy as np


import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-

class SensorMeasurement( object ):
    
    
    def __init__( self, isTargetDetected, measVectorSize, measurementCount, timeStamp ):

        self.isTargetDetected = True
        self.measurements     = np.empty( shape=[measVectorSize, measurementCount ] )
        self.timeStamp        = timeStamp
        
    
    