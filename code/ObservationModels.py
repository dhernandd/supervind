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

# The lines below were first seen in the walls of Alcatraz.
if __name__ == 'ObservationModels':
    from layers import FullLayer  # @UnresolvedImport @UnusedImport
else:
    from .layers import FullLayer  # @Reimport

TEST_DIR = './tests/test_results/'

DTYPE = tf.float32

class ObsModel():
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
        
    def compute_LogDensity(self):
        """
        """
        raise NotImplementedError("This is an abstract method. Please define it in "
                                  "the child classes")

    def sample_XY(self):
        """
        """
        raise NotImplementedError("This is an abstract method. Please define it in "
                                  "the child classes")


class PoissonObs(ObsModel):
    """
    """
    def __init__(self, Y, X, params, lat_ev_model, is_out_positive=False):
        """
        """
        ObsModel.__init__(self, Y, X, params, lat_ev_model, is_out_positive)

        self.rate_NTxD = self._define_rate()
        self.LogDensity, self.checks = self.compute_LogDensity() # self.checks meant for debugging
    
    def _define_rate(self, Input=None):
        """
        """
        if Input is None: Input = self.X
        
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
        
        checks = [LY, LY1, LY2, LY3, LX]
        checks.extend(Xchecks)
        
        return tf.add(LX, LY, name='LogDensity'), checks 


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
        rate = np.reshape(rate, [Nsamps, NTbins, self.yDim])
        Ydata_NxTxD = np.random.poisson(rate)
        
        return Ydata_NxTxD, Xdata_NxTxd
    
    
class GaussianObs():
    """
    """
    def __init__(self, Y, X, params, lat_ev_model, is_out_positive=False):
        """
        """
        ObsModel.__init__(self, Y, X, params, lat_ev_model, is_out_positive)
        
        self.MuY_NxTxD, self.SigmaInvY_DxD = self._define_mean_variance()
        self.LogDensity, self.checks = self.compute_LogDensity() # self.checks meant for debugging
    
    def _define_mean_variance(self, Input=None):
        """
        """
        if Input is None: Input = self.X
        
        xDim = self.xDim
        yDim = self.yDim
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]

        Input = tf.reshape(Input, [Nsamps*NTbins, xDim], name='X_input')
        
        rangeY = self.params.initrange_Goutmean
        initSigma = self.params.initrange_Goutvar
        init_b = self.params.initbias_Goutmean
        obs_nodes = 64
        fully_connected_layer = FullLayer()
        with tf.variable_scope("obs_nn_mean", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input, obs_nodes, 'softplus', 'full1',
                                          initializer=tf.random_normal_initializer(stddev=0.5))
            full2 = fully_connected_layer(full1, obs_nodes, 'softplus', 'full2',
                                          initializer=tf.random_normal_initializer(stddev=0.5))
            MuY_NTxD = fully_connected_layer(full2, yDim, 'linear', 'output',
                                             initializer=tf.random_uniform_initializer(-rangeY, rangeY),
                                             b_initializer=tf.random_normal_initializer(init_b) )
            MuY_NxTxD = tf.reshape(MuY_NTxD, [Nsamps, NTbins, yDim])
        with tf.variable_scope("obs_var", reuse=tf.AUTO_REUSE):
            SigmaInvChol_DxD = tf.get_variable('SigmaInvChol', 
                                                initializer=tf.cast(initSigma*tf.eye(yDim), DTYPE))
            self.SigmaChol_1x1xDxD = tf.reshape(tf.matrix_inverse(SigmaInvChol_DxD),
                                                [1, 1, yDim, yDim]) # Needed only for sampling
            SigmaInv_DxD = tf.matmul(SigmaInvChol_DxD, SigmaInvChol_DxD,
                                        transpose_b=True)
            
        return MuY_NxTxD, SigmaInv_DxD 
        
    def compute_LogDensity(self, Input=None, with_inflow=False):
        """
        """
        yDim = self.yDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            X = self.X
            LX, checks = self.lat_ev_model.compute_LogDensity_Xterms(with_inflow=with_inflow) # checks=[LX0, LX1, LX2, LX3, LX4]
            MuY_NxTxD, SigmaInvY_DxD = self.MuY_NxTxD, self.SigmaInvY_DxD
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            X = Input
            LX, checks = self.lat_ev_model.compute_LogDensity_Xterms(X, with_inflow=with_inflow) # checks=[LX0, LX1, LX2, LX3, LX4]
            MuY_NxTxD, SigmaInvY_DxD = self._define_mean_variance(X)
        
        SigmaInvY_NTxDxD = tf.tile(tf.expand_dims(SigmaInvY_DxD, axis=0), [Nsamps*NTbins, 1, 1])
        MuY_NTx1xD = tf.reshape(MuY_NxTxD, [Nsamps*NTbins, 1, yDim])
        Y_NTx1xD = tf.reshape(self.Y, [Nsamps*NTbins, 1, yDim])
        
        DeltaY_NTx1xD = Y_NTx1xD - MuY_NTx1xD
        
        LY1 = -0.5*tf.reduce_sum(DeltaY_NTx1xD*tf.matmul(DeltaY_NTx1xD, SigmaInvY_NTxDxD))
        LY2 = 0.5*tf.reduce_sum(tf.log(tf.matrix_determinant(SigmaInvY_DxD)))*tf.cast(Nsamps*NTbins, DTYPE)
        LY = LY1 + LY2
        
        checks.extend([LX, LY, LY1, LY2])
        return tf.add(LX, LY, name='LogDensity'), checks

    #** These methods take a session as input and are not part of the main
    #** graph. They are meant to be used as standalone.
    
    def sample_XY(self, sess, Xvar_name, Nsamps=50, NTbins=100, X0data=None, 
                 with_inflow=True, path_mse_threshold=1.0,
                 draw_plots=False, init_variables=False):
        """
        """
        yDim = self.yDim
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd = self.lat_ev_model.sample_X(sess, Xvar_name, Nsamps=Nsamps, NTbins=NTbins,
                                           X0data=X0data, with_inflow=with_inflow, 
                                           path_mse_threshold=path_mse_threshold, 
                                           draw_plots=draw_plots,
                                           init_variables=init_variables)
        
        MuY_NxTxD = self.MuY_NxTxD
        SigmaChol_NxTxDxD = tf.tile(self.SigmaChol_1x1xDxD, [Nsamps, NTbins, 1, 1])
        noise_NxTx1xD = tf.random_normal([Nsamps, NTbins, 1, yDim])
        
        sampleY_NxTxD = MuY_NxTxD + tf.reshape(tf.matmul(noise_NxTx1xD, SigmaChol_NxTxDxD),
                                               [Nsamps, NTbins, yDim])
        Ydata_NxTxD = sess.run(sampleY_NxTxD, feed_dict={Xvar_name : Xdata_NxTxd})
        
        return Ydata_NxTxD, Xdata_NxTxd



class CellVoltageObs(ObsModel):
    """
    """
    def __init__(self, Y, X, params, lat_ev_model, is_out_positive=False):
        """
        """
        ObsModel.__init__(self, Y, X, params, lat_ev_model, is_out_positive)
        if self.yDim != 1:
            raise ValueError("The data dimension must be 1 for cell voltage data")
        
        self.MuY_NxTx1, self.sigmaInvY = self._define_mean_variance()
        self.LogDensity, self.checks = self.compute_LogDensity() # self.checks meant for debugging
    
    def _define_mean_variance(self, Input=None):
        """
        """
        if Input is None: Input = self.X
        
        Input_dxNxT = tf.transpose(Input, [2, 0, 1])
        Input_dxxNxT = tf.unstack(Input_dxNxT)
        MuY_NxTx1 = tf.expand_dims(Input_dxxNxT[0], axis=2)
        
        with tf.variable_scope("obs_var", reuse=tf.AUTO_REUSE):
            sigmaInvChol = tf.get_variable('SigmaInvChol', initializer=tf.cast(1.0, DTYPE))
            self.sigmaChol = 1.0/sigmaInvChol # Needed only for sampling
            sigmaInv = sigmaInvChol**2
            
        return MuY_NxTx1, sigmaInv

    def compute_LogDensity(self, Input=None, Ids=None, with_inflow=False):
        """
        """
        latm = self.lat_ev_model
        # TODO: Define behavior when Ids are given
        if Input is None:
            X_NxTxd = self.X
            Nsamps = tf.shape(X_NxTxd)[0]
            NTbins = tf.shape(X_NxTxd)[1]
            
            if Ids is None:
                Ids = latm.Ids
                LX, checks = latm.logdensity_Xterms, latm.checks_LX # checks = [LX0, LX1, LX2, LX3, LX4]
            else:
                LX, checks = self.lat_ev_model.compute_LogDensity_Xterms(X_NxTxd, Ids,
                                                                         with_inflow=with_inflow)
            MuY_NxTx1, sigmaInvY = self.MuY_NxTx1, self.sigmaInvY
        else:
            X_NxTxd = Input
            Nsamps = tf.shape(X_NxTxd)[0]
            NTbins = tf.shape(X_NxTxd)[1]
            
            # checks = [LX0, LX1, LX2, LX3, LX4]
            if Ids is None:
                raise ValueError("You must provide Ids for this Input")
            else:
                LX, checks = self.lat_ev_model.compute_LogDensity_Xterms(X_NxTxd, Ids,
                                                                         with_inflow=with_inflow)
            MuY_NxTx1, sigmaInvY = self._define_mean_variance(X_NxTxd)
        
        MuY_NTx1x1 = tf.reshape(MuY_NxTx1, [Nsamps*NTbins, 1, 1])
        Y_NTx1x1 = tf.reshape(self.Y, [Nsamps*NTbins, 1, 1])
        
        DeltaY_NT = tf.squeeze(Y_NTx1x1 - MuY_NTx1x1)
        
        LY1 = -0.5*tf.reduce_sum(DeltaY_NT*DeltaY_NT*sigmaInvY)
        LY2 = 0.5*tf.reduce_sum(tf.log(sigmaInvY))*tf.cast(Nsamps*NTbins, DTYPE)
        LY = LY1 + LY2
        
        checks.extend([LX, LY, LY1, LY2])
        return tf.add(LX, LY, name='LogDensity'), checks # checks = [LX0, LX1, LX2, LX3, LX4, LX, LY, LY1, LY2]

    def sample_XY(self, sess, prefix, Ids=None, Nsamps=50, NTbins=100, X0data=None, 
                 with_inflow=True, path_mse_threshold=1.0,
                 draw_plots=False, init_variables=False):
        """
        """
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd, Ids_N = self.lat_ev_model.sample_X(sess, prefix, Ids=Ids, Nsamps=Nsamps,
                                                        NTbins=NTbins,
                                                        X0data=X0data, with_inflow=with_inflow, 
                                                        path_mse_threshold=path_mse_threshold, 
                                                        draw_plots=draw_plots)

        MuY_NxTx1 = self.MuY_NxTx1
        sigmaChol = self.sigmaChol
        noise_NxTx1 = tf.random_normal([Nsamps, NTbins, 1])

        sampleY_NxTx1 = MuY_NxTx1 + noise_NxTx1*sigmaChol
        Ydata_NxTx1 = sess.run(sampleY_NxTx1, feed_dict={prefix+'X:0' : Xdata_NxTxd})
        
        return Ydata_NxTx1, Xdata_NxTxd, Ids_N


