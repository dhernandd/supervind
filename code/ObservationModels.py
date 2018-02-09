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
    nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity}
    nonlinearity = nl_dict[nl]
    
    weights_full1 = variable_in_cpu('weights', [input_dim, nodes], 
                          initializer=tf.random_normal_initializer())
    biases_full1 = variable_in_cpu('biases', [nodes], 
                             initializer=tf.constant_initializer())
    full = nonlinearity(tf.matmul(Input, weights_full1) + biases_full1,
                          name='full1')
    return full


class PoissonObs():
    """
    """
    def __init__(self, yDim, xDim, Y, X, lat_ev_model, is_out_positive=False):
        """
        """
        self.yDim = yDim
        self.xDim = xDim
        self.Y = Y
        self.X = X
        self.lat_ev_model = lat_ev_model
        self.is_out_positive = is_out_positive
        
        self.Nsamps = Nsamps = tf.shape(self.X)[0]
        self.NTbins = NTbins = tf.shape(self.X)[1]

        obs_nodes = 64
        X_input_NTxd = tf.reshape(self.X, [Nsamps*NTbins, xDim], name='X_input')
        with tf.variable_scope("Observation_Network"):
            with tf.variable_scope('full1'):
                full1 = FullLayer(X_input_NTxd, obs_nodes, xDim, 'softplus')
            with tf.variable_scope('full2'):
                full2 = FullLayer(full1, obs_nodes, obs_nodes, 'softplus')
            with tf.variable_scope('full3'):
                full3 = FullLayer(full2, yDim, obs_nodes, 'linear')
        self.inv_tau = 0.002
        self.rate_NTxD = full3 if self.is_out_positive else tf.exp(self.inv_tau*full3) 

    
    def compute_LogDensity_Yterms(self):
        Y_NTxD = tf.reshape(self.Y, [self.Nsamps*self.NTbins, self.yDim])
        return tf.reduce_sum(Y_NTxD*tf.log(self.rate_NTxD) - self.rate_NTxD -
                             tf.lgamma(Y_NTxD + 1.0))
        
    def compute_LogDensity(self):
        LX, _ = self.lat_ev_model.compute_LogDensity_Xterms()
        LY = self.compute_LogDensity_Yterms()
        return LX + LY

    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used as standalone.
    
    def sample_XY(self, sess, Nsamps=50, NTbins=100, X0data=None, inflow_scale=0.9, 
                 with_inflow=False, path_mse_threshold=1.0, init_from_save=False,
                 draw_plots=False, init_variables=True):
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
                                           init_variables=init_variables)
        
#         print('Xdata_NxTxd:', Xdata_NxTxd)
        rate = sess.run(self.rate_NTxD, feed_dict={'X:0' : Xdata_NxTxd})
        rate = np.reshape(rate, [Nsamps, NTbins, self.yDim])
#         print('rate:', rate)
        Ydata_NxTxD = np.random.poisson(rate)
        
        return Ydata_NxTxD, Xdata_NxTxd

