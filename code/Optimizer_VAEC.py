# Copyright 2017 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from LatEvModels import LocallyLinearEvolution
from ObservationModels import PoissonObs
from RecognitionModels import SmoothingNLDSTimeSeries


class Optimizer_TS():
    """
    """
    def __init__(self, yDim, xDim, EvolutionModel=LocallyLinearEvolution,
                 ObsModel=PoissonObs, RecModel=SmoothingNLDSTimeSeries):
        """
        """
#         Trainable.__init__(self)
        
        self.xDim = xDim
        self.yDim = yDim
        
        self.graph = graph = tf.Graph()
        with self.graph.as_default():
            self.X = X = tf.placeholder(tf.float64, [None, None, self.xDim], name='X')
            self.Y = Y = tf.placeholder(tf.float64, [None, None, self.yDim], name='Y')
            
            self.lat_ev_model = lat_ev_model = EvolutionModel(xDim, X)
            self.mgen = mgen = ObsModel(yDim, xDim, Y, X, lat_ev_model)
            self.mrec = mrec = SmoothingNLDSTimeSeries(yDim, xDim, Y, X, lat_ev_model)
            
#         LatPars = ParsDicts['LatPars']
#         self.common_lat = common_lat
#         if common_lat:
        
#         else:
#             lat_ev_model = None
        
#         ObsPars = ParsDicts['ObsPars']
#         GENLATCLASS = ObsPars['LATCLASS'] if 'LATCLASS' in ObsPars else None
#         self.mgen = OBSCLASS(ObsPars, yDim, xDim, Y, X, lat_ev_model=lat_ev_model, LATCLASS=GENLATCLASS)
#         
#         RecPars = ParsDicts['RecPars']
#         RECLATCLASS = RecPars['LATCLASS'] if 'LATCLASS' in RecPars else None
#         if RECCLASS is not None: self.mrec = RECCLASS(RecPars, yDim, xDim, Y, X, lat_ev_model, LATCLASS=RECLATCLASS)

 
 