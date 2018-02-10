# Copyright 2018 Daniel Hernandez Diaz, Columbia University
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
from RecognitionModels_new import SmoothingNLDSTimeSeries


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
        
#         self.graph = graph = tf.Graph()
        with tf.Graph().as_default() as grec:
            self.Y = Y = tf.placeholder(tf.float64, [None, None, self.yDim], name='Y')
            self.mrec = mrec = RecModel(yDim, xDim, Y)
#             self.X = X = tf.placeholder(tf.float64, [None, None, self.xDim], name='X')
#             self.Y = Y = tf.placeholder(tf.float64, [None, None, self.yDim], name='Y')
#             
#             self.lat_ev_model = lat_ev_model = EvolutionModel(xDim, X)
            self.mgen = mgen = ObsModel(yDim, xDim, Y, X, lat_ev_model)
#             
#             self.graph_def = graph.as_graph_def()
            
            
    
#     def cost_ELBO(self):
# #         Nsamps = self.mgen.Nsamps
#         
#         postX = self.mrec.noisy_postX
#         
#         LogDensity = tf.import_graph_def(self.graph_def, 
#                                          input_map={'X:0' : postX}, 
#                                          return_elements=["LogDensity:0"])
#         
#         return LogDensity
#         Entropy = self.mrec.compute_Entropy()
        
#         Nsamps = Y.shape[0]
#         LogDensity = mgen.compute_LogDensity(Y, postX, padleft=padleft) 
#         Entropy = mrec.compute_Entropy(Y, postX)
#         ELBO = (LogDensity + Entropy if not regularize_evolution_weights else 
#                 LogDensity + Entropy + lat_weights_regloss)
#         costs_func = theano.function(inputs=self.CostsInputDict['ELBO'], 
#                                      outputs=[ELBO/Nsamps, LogDensity/Nsamps, Entropy/Nsamps])

 