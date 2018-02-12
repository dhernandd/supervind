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

from LatEvModels_new import LocallyLinearEvolution
from ObservationModels_new import PoissonObs
from RecognitionModels_new import SmoothingNLDSTimeSeries


class Optimizer_TS():
    """
    """
    def __init__(self, yDim, xDim, ObsModel=PoissonObs, 
                 RecModel=SmoothingNLDSTimeSeries, learning_rate=1e-3):
        """
        """
#         Trainable.__init__(self)
        
        self.xDim = xDim
        self.yDim = yDim
        self.learning_rate = lr = learning_rate
        
        self.graph = graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('VAEC', reuse=tf.AUTO_REUSE):
                self.Y = Y = tf.placeholder(tf.float64, [None, None, self.yDim], name='Y')
                self.X = X = tf.placeholder(tf.float64, [None, None, self.xDim], name='X')
                self.mrec = mrec = RecModel(yDim, xDim, Y, X)
    #             
                self.lat_ev_model = lat_ev_model = self.mrec.lat_ev_model
<<<<<<< HEAD
                self.mgen = mgen = ObsModel(yDim, xDim, Y, X, lat_ev_model)
                
                self.cost = self.cost_ELBO()
                self.cost_with_inflow = self.cost_ELBO(with_inflow=True)
                
                self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope=tf.get_variable_scope().name)
                for i in range(len(self.train_vars)):
                    shape = self.train_vars[i].get_shape().as_list()
                    print("    ", i, self.train_vars[i].name, shape)
                
                grads = tf.gradients(self.cost, self.train_vars)
                opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999,
                                 epsilon=1e-01)
                self.grads = grads
#                 self.grad_global_norm = grad_global_norm
                self.train_step = tf.get_variable("global_step", [], tf.int64,
                                                  tf.zeros_initializer(),
                                                  trainable=False)
                self.train_op = opt.apply_gradients(zip(grads, self.train_vars),
                                                    global_step=self.train_step)

    
    def cost_ELBO(self, with_inflow=False):
         
        postX = self.mrec.postX
        LogDensity, _, _ = self.mgen.compute_LogDensity(postX, with_inflow=with_inflow)
        Entropy = self.mrec.compute_Entropy(postX)
        
<<<<<<< HEAD
        return LogDensity
=======
                self.mgen = mgen = ObsModel(yDim, xDim, X, lat_ev_model)
#             
#             self.graph_def = graph.as_graph_def()
            
            
>>>>>>> 4c5c36502666a34bd96835f1269d133585771520
=======
        return -(LogDensity + Entropy)
    
>>>>>>> develop
    
        
#         Nsamps = Y.shape[0]
#         LogDensity = mgen.compute_LogDensity(Y, postX, padleft=padleft) 
#         Entropy = mrec.compute_Entropy(Y, postX)
#         ELBO = (LogDensity + Entropy if not regularize_evolution_weights else 
#                 LogDensity + Entropy + lat_weights_regloss)
#         costs_func = theano.function(inputs=self.CostsInputDict['ELBO'], 
#                                      outputs=[ELBO/Nsamps, LogDensity/Nsamps, Entropy/Nsamps])

    def train_epoch(self, Ydata, Xdata):
        """
        """
        session = tf.get_default_session()
        train = session.run([self.train_op, self.cost], 
                            feed_dict={'VAEC/X:0' : Xdata,
                                       'VAEC/Y:0' : Ydata})
        print('Cost:', train[1])
        
    
    def train(self, Ydata, num_epochs=20):
        """
        """
        Ydata_NxTxD = Ydata
        started_training = False
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(num_epochs):
                if not started_training:
                    Xpassed_NxTxd = sess.run(self.mrec.Mu_NxTxd, 
                                             feed_dict={'VAEC/Y:0' : Ydata_NxTxD}) 
                    started_training = True
                else:
                    Xpassed_NxTxd = sess.run(self.mrec.postX, 
                                             feed_dict={'VAEC/Y:0' : Ydata_NxTxD,
                                                        'VAEC/X:0' : Xpassed_NxTxd})
                self.train_epoch(Ydata, Xpassed_NxTxd)
            
        
        
    
    
    
    
    
    