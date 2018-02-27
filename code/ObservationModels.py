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
# The lines below were first written in the walls of Alcatraz.
if __name__ == 'ObservationModels':
    from layers import FullLayer  # @UnresolvedImport @UnusedImport
else:
    from .layers import FullLayer  # @Reimport

TEST_DIR = '/Users/danielhernandez/work/supervind/tests/test_results/'


class PoissonObs():
    """
    """
    def __init__(self, Y, X, params, lat_ev_model, is_out_positive=False):
        """
        """
        self.X = X
        self.params = params

        self.yDim = params.yDim
        self.xDim = params.xDim
        self.lat_ev_model = lat_ev_model
        self.is_out_positive = is_out_positive
        self.Y = Y
        
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]
        
        self.rate_NTxD = self._define_rate(X)
        self.LogDensity, self.checks = self.compute_LogDensity() # self.checks meant for debugging
    
    def _define_rate(self, Input):
        """
        """
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]
        xDim = self.xDim
        yDim = self.yDim
        Input = tf.reshape(Input, [Nsamps*NTbins, xDim], name='X_input')
        
        rangeY = self.params.initrange_outY
        self.inv_tau = inv_tau = 0.3
        obs_nodes = 64
        fully_connected_layer = FullLayer()
        with tf.variable_scope("obs_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input, obs_nodes, 'softplus', 'full1')
            full2 = fully_connected_layer(full1, obs_nodes, 'softplus', 'full2')
            if self.is_out_positive:
                rate_NTxD = fully_connected_layer(full2, yDim, 'softplus', 'output',
                                                  b_initializer=tf.random_normal_initializer(1.0, rangeY))
            else:
                full3 = fully_connected_layer(full2, yDim, 'linear', 'output',
                                              initializer=tf.random_uniform_initializer(-rangeY, rangeY))
                rate_NTxD = tf.exp(inv_tau*full3)
        
        return rate_NTxD
        
    def compute_LogDensity(self, Input=None, with_inflow=False):
        """
        """
        yDim = self.yDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            X = self.X
            LX, Xchecks = self.lat_ev_model.compute_LogDensity_Xterms(with_inflow=with_inflow)
            rate_NTxD = self.rate_NTxD
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            X = Input
            LX, Xchecks = self.lat_ev_model.compute_LogDensity_Xterms(X, 
                                                                with_inflow=with_inflow)        
            rate_NTxD = tf.identity(self._define_rate(X), name='rate_'+X.name[:-2])
        
        Y_NTxD = tf.reshape(self.Y, [Nsamps*NTbins, yDim])
        LY1 = tf.reduce_sum(Y_NTxD*tf.log(rate_NTxD))
        LY2 = tf.reduce_sum(-rate_NTxD)
        LY3 = tf.reduce_sum(- tf.lgamma(Y_NTxD + 1.0))
        LY = LY1 + LY2 + LY3
        
        tf.summary.scalar('LogDensity_Yterms', LY) 
        self.LY1_summ = tf.summary.scalar('LY1', LY1)
        
        return tf.add(LX, LY, name='LogDensity'), [LY, LY1, LY2, LY3, LX, *tuple(Xchecks)]


    #** These methods take a session as input and are not part of the main
    #** graph. They are meant to be used as standalone.
    
    def sample_XY(self, sess, Xvar_name, Nsamps=50, NTbins=100, X0data=None, 
                 with_inflow=True, path_mse_threshold=1.0,
                 draw_plots=False, init_variables=False):
        """
        """
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd = self.lat_ev_model.sample_X(sess, Xvar_name, Nsamps=Nsamps, NTbins=NTbins,
                                           X0data=X0data, with_inflow=with_inflow, 
                                           path_mse_threshold=path_mse_threshold, 
                                           draw_plots=draw_plots,
                                           init_variables=init_variables)
        
        rate_NTxD = self.rate_NTxD
        rate = sess.run(rate_NTxD, feed_dict={Xvar_name : Xdata_NxTxd})
        print('Sampled rate (mean, std, max)\n', np.mean(rate), np.std(rate),
              np.max(rate))

        rate = np.reshape(rate, [Nsamps, NTbins, self.yDim])
        Ydata_NxTxD = np.random.poisson(rate)
        print('Ydata (mean, std, max)\n', np.mean(Ydata_NxTxD), np.std(Ydata_NxTxD),
              np.max(Ydata_NxTxD))
        
        return Ydata_NxTxD, Xdata_NxTxd
    
    
class GaussianObs():
    pass



