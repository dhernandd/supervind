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

from utils import variable_in_cpu

TEST_DIR = '/Users/danielhernandez/work/supervind/tests/test_results/'

def FullLayer(Input, nodes, input_dim=None, nl='softplus'):
    """
    """
    nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity}
    nonlinearity = nl_dict[nl]
    
    weights_full = variable_in_cpu('weights', [input_dim, nodes],
                                   initializer=tf.orthogonal_initializer())
    biases_full = variable_in_cpu('biases', [nodes],
                                  initializer=tf.zeros_initializer(dtype=tf.float64))
    full = nonlinearity(tf.matmul(Input, weights_full) + biases_full,
                          name='output')
    return full


class PoissonObs():
    """
    """
    def __init__(self, yDim, xDim, Y, X, lat_ev_model, is_out_positive=False):
        """
        """
        self.yDim = yDim
        self.xDim = xDim
        self.X = X
        self.lat_ev_model = lat_ev_model
        self.is_out_positive = is_out_positive
        self.Y = Y
        
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]
        
        self.rate_NTxD = self._define_rate(X)
#         self.LogDensity = self.compute_LogDensity() 

    
    def _define_rate(self, Input):
        """
        """
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]
        xDim = self.xDim 
        Input = tf.reshape(Input, [Nsamps*NTbins, xDim], name='X_input')
        
        self.inv_tau = inv_tau = 0.002
        obs_nodes = 64
        with tf.variable_scope("obs_nn", reuse=tf.AUTO_REUSE):
            with tf.variable_scope('full1'):
                full1 = FullLayer(Input, obs_nodes, xDim, 'softplus')
            with tf.variable_scope('full2'):
                full2 = FullLayer(full1, obs_nodes, obs_nodes, 'softplus')
            with tf.variable_scope('full3'):
                full3 = FullLayer(full2, self.yDim, obs_nodes, 'linear')
        rate_NTxD = full3 if self.is_out_positive else tf.exp(inv_tau*full3)
        
        return rate_NTxD
        
#     
#     def compute_LogDensity_Yterms(self, Y):
#         Y_NTxD = tf.reshape(self.Y, [self.Nsamps*self.NTbins, self.yDim])
#         return tf.reduce_sum(Y_NTxD*tf.log(self.rate_NTxD) - self.rate_NTxD -
#                              tf.lgamma(Y_NTxD + 1.0))
        
    def compute_LogDensity(self, X, with_inflow=False):
        """
        """
        Nsamps = tf.shape(X)[0]
        NTbins = tf.shape(X)[1]
        yDim = self.yDim
        
        LX, _ = self.lat_ev_model.compute_LogDensity_Xterms(X, with_inflow)

        rate_NTxD = tf.identity(self._define_rate(X), name='rate')
        Y_NTxD = tf.reshape(self.Y, [Nsamps*NTbins, yDim])
        LY = tf.reduce_sum(Y_NTxD*tf.log(rate_NTxD) - rate_NTxD -
                         tf.lgamma(Y_NTxD + 1.0))
        
        return tf.add(LX, LY, name='LogDensity'), LX, LY


    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used as standalone.
    
    def sample_XY(self, sess, Nsamps=50, NTbins=100, X0data=None, inflow_scale=0.9, 
                 with_inflow=False, path_mse_threshold=1.0, init_from_save=False,
                 draw_plots=False, init_variables=True, feed_key='X:0'):
        """
        """
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd = self.lat_ev_model.sample_X(sess, Nsamps=Nsamps, NTbins=NTbins,
                                           X0data=X0data, inflow_scale=inflow_scale, 
                                           with_inflow=with_inflow, 
                                           path_mse_threshold=path_mse_threshold, 
                                           init_from_save=init_from_save, 
                                           draw_plots=draw_plots,
                                           init_variables=init_variables,
                                           feed_key=feed_key)
        
#         Input = tf.reshape(self.X, [self.Nsamps, self.NTbins, self.xDim])
        rate_NTxD = self.rate_NTxD
        rate = sess.run(rate_NTxD, feed_dict={feed_key : Xdata_NxTxd})
        rate = np.reshape(rate, [Nsamps, NTbins, self.yDim])
        Ydata_NxTxD = np.random.poisson(rate)
        
        return Ydata_NxTxD, Xdata_NxTxd

