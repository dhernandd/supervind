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
import os
# import time

import numpy as np

import tensorflow as tf

from code.datetools import addDateTime
from code.layers import FullLayer
from code.LatEvModels import NoisyEvolution

TEST_DIR = '/Users/danielhernandez/work/supervind/tests/test_results/'
DTYPE = tf.float32

def flow_modulator(x, x0=30.0, a=0.08):
    return (1 - np.tanh(a*(x - x0)))/2

def flow_modulator_tf(X, x0=30.0, a=0.08):
    return tf.cast((1.0 - tf.tanh(a*(X - x0)) )/2, DTYPE)


class NoisyEvolution_wParams():
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, X, params):
        """
        Args:
            X:
            Ids: List of size Nsamps of identities for each trial. 
        """
        self.X = X
        self.params = params
        
        self.xDim = xDim = params.xDim
        self.pDim = pDim = params.pDim
        self.rDim = xDim + pDim
        self.X = X
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]

        if hasattr(params, 'num_diff_entities'): self.num_diff_entities = params.num_diff_entities
        else: self.num_diff_entities = 1
        self.ev_params_Pxp = tf.get_variable('ev_params', shape=[self.num_diff_entities, pDim])
        
        # Variance of the state-space evolution. Assumed to be constant
        # throughout.
        if not hasattr(self, 'QInvChol'):
            self.QInvChol_dxd = tf.get_variable('QInvChol', 
                                initializer=tf.cast(tf.eye(xDim), DTYPE))
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                              name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True,
                           name='Q')
        
        # Variance of the initial points
        if not hasattr(self, 'Q0InvChol'):
            self.Q0InvChol_dxd = tf.get_variable('Q0InvChol', 
                                initializer=tf.cast(tf.eye(xDim), DTYPE))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd,
                                   transpose_b=True, name='Q0Inv')
        
        # The starting coordinates in state-space
        self.x0 = tf.get_variable('x0', 
                                  initializer=tf.cast(tf.zeros(self.xDim), DTYPE) )
        
        if not hasattr(self, 'alpha'):
            init_alpha = tf.constant_initializer(0.2)
            self.alpha = tf.get_variable('alpha', shape=[], 
                                                 initializer=init_alpha,
                                                 dtype=DTYPE)

#       Define base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear_dxd = tf.get_variable('Alinear', initializer=tf.eye(xDim),
                                               dtype=DTYPE)

        self.A_NxTxdxd, self.Awinflow_NxTxdxd, self.B_NxTxdxd = self._define_evolution_network()
        
        # The gradients yeah?
#         self.x = tf.placeholder(dtype=tf.float64, shape=[1, 1, xDim], name='x')
#         self.Agrads_r2xd = self.get_A_grads()
        
    def _define_evolution_network(self, Input=None, Ids=None):
        """
        """
        xDim = self.xDim
        pDim = self.pDim
        rDim = xDim + pDim
        
        Input_NxTxd = self.X if Input is None else Input
        Nsamps = Input_NxTxd.get_shape().as_list()[0]
        NTbins = tf.shape(Input_NxTxd)[1]
        if Ids is None:
            Ids = [0]*Nsamps # If no Ids provided, assume the same Id.
        else:
            pass # TODO: check that Ids is compatible with self.ev_params!

        ev_params_Nxp = []
        ev_params_unstacked = tf.unstack(self.ev_params_Pxp)
        for idx in Ids:
            ev_params_Nxp.append(ev_params_unstacked[idx])
        ev_params_Nxp = tf.stack(ev_params_Nxp)
        ev_params_NxTxp = tf.tile(tf.reshape(ev_params_Nxp, [Nsamps, 1, pDim]), [1, NTbins, 1])

        Input_NxTxr = tf.concat([Input_NxTxd, ev_params_NxTxp], axis=2)
            
        alpha = self.alpha
        evnodes = 64
        Input_NTxr = tf.reshape(Input_NxTxr, [Nsamps*NTbins, rDim])
        fully_connected_layer = FullLayer()
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input_NTxr, evnodes, 'softmax', 'full1')
            output = fully_connected_layer(full1, xDim**2, 'linear', 'output')
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim], name='B')
        
        # Broadcast
        A_NTxdxd = alpha*B_NTxdxd + self.Alinear_dxd
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim], name='A')
        
        X_norms = tf.norm(Input_NTxr[:,:xDim], axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        eye_swap = tf.transpose(tf.tile(tf.expand_dims(tf.eye(self.xDim), 0),
                                        [Nsamps*NTbins, 1, 1]), [2,1,0])
        Awinflow_NTxdxd = tf.transpose(fl_mod*tf.transpose(
            A_NTxdxd, [2,1,0]) + 0.9*(1.0-fl_mod)*eye_swap, [2,1,0])
        
        Awinflow_NxTxdxd = tf.reshape(Awinflow_NTxdxd, 
                                      [Nsamps, NTbins, xDim, xDim], name='Awinflow')
        
        return A_NxTxdxd, Awinflow_NxTxdxd, B_NxTxdxd

    def get_A_grads(self, xin=None, ids=None):
        xDim = self.xDim
        if xin is None: xin = self.x

        singleA_1x1xdxd = self._define_evolution_network(xin, Ids=ids)[0]
        singleA_d2 = tf.reshape(singleA_1x1xdxd, [xDim**2])
        grad_list_d2xd = tf.squeeze(tf.stack([tf.gradients(Ai, xin) for Ai
                                              in tf.unstack(singleA_d2)]))

        return grad_list_d2xd 
        
    
    
class LocallyLinearEvolution_wParams(NoisyEvolution_wParams):
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, xDim, X, pDim=0):
        """
        """
        NoisyEvolution_wParams.__init__(self, xDim, X, pDim)
                        
        self.logdensity_Xterms = self.compute_LogDensity_Xterms(with_inflow=True)
        
        
    def compute_LogDensity_Xterms(self, Input=None, with_inflow=False):
        """
        Computes the symbolic log p(X, Y).
        p(X, Y) is computed using Bayes Rule. p(X, Y) = P(Y|X)p(X).
        p(X) is normal as described in help(PNLDS).
        p(Y|X) is py with output self.output(X).
         
        Inputs:
            X : Symbolic array of latent variables.
            Y : Symbolic array of X
         
        NOTE: This function is required to accept symbolic inputs not necessarily belonging to the class.
        """
        xDim = self.xDim
        rDim = self.rDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            totalA_NxTxrxr = ( self.A_NxTxrxr if not with_inflow else 
                               self.Awinflow_NxTxrxr )
            X = self.X
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            A_NxTxrxr, Awinflow_NTxrxr = self._define_evolution_network(Input)
            totalA_NxTxrxr = A_NxTxrxr if not with_inflow else Awinflow_NTxrxr
            X = Input

        totalA_NTm1xrxr = tf.reshape(totalA_NxTxrxr[:,:-1,:,:], 
                                     [Nsamps*(NTbins-1), rDim, rDim])
        Xin_NTm1x1xr = tf.reshape(X[:,:-1,:rDim], [Nsamps*(NTbins-1), 1, rDim])
        Xprime_NTm1xr = tf.reshape(tf.matmul(Xin_NTm1x1xr, totalA_NTm1xrxr), 
                                    [Nsamps*(NTbins-1), rDim])

        resX_NTm1xr = ( tf.reshape(X[:,1:,:rDim], [Nsamps*(NTbins-1), rDim])
                                    - Xprime_NTm1xr )
        resX0_Nxr = X[:,0,:rDim] - self.x0[:rDim]
        
        # L = -0.5*(∆X_0^T·Q0^{-1}·∆X_0) - 0.5*Tr[∆X^T·Q^{-1}·∆X] + 0.5*N*log(Det[Q0^{-1}])
        #     + 0.5*N*T*log(Det[Q^{-1}]) - 0.5*N*T*d_X*log(2*Pi)
        L1 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX0_Nxr, self.Q0Inv_rxr), 
                             resX0_Nxr, transpose_b=True), name='L1') 
        L2 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX_NTm1xr, self.QInv_rxr), 
                             resX_NTm1xr, transpose_b=True), name='L2' )
        L3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_rxr))*tf.cast(Nsamps, tf.float64)
        L4 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_rxr))*
               tf.cast((NTbins-1)*Nsamps, tf.float64) )
        L5 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*rDim, tf.float64)
        
        LatentDensity = L1 + L2 + L3 + L4 + L5
                
        return LatentDensity, [L1, L2, L3, L4, L5]
    
    
    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, Nsamps=2, NTbins=3, X0data=None, inflow_scale=0.9, 
                 with_inflow=False, path_mse_threshold=1.0, init_from_save=False,
                 draw_plots=False, init_variables=True, feed_key='X:0'):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        if init_variables: 
            sess.run(tf.global_variables_initializer())
        
        xDim = self.xDim
        rDim = self.rDim
        
        Q0Chol = sess.run(self.Q0Chol_rxr)
        QChol = sess.run(self.QChol_rxr)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, self.xDim])
        x0scale = 25.0
        
        A_NxTxrxr = self.Awinflow_NxTxrxr if with_inflow else self.A_NxTxrxr
        A_NTxrxr = tf.reshape(A_NxTxrxr, shape=[-1, rDim, rDim])
        for samp in range(Nsamps):
            # needed to avoid paths that start too close to an attractor
            samp_norm = 0.0
            
            # lower path_mse_threshold to keep paths closer to trivial
            # trajectories, x = const.
            while samp_norm < path_mse_threshold:
                X_single_samp_1xTxd = np.zeros([1, NTbins, self.xDim])
                
                Q0Chol_dxd = np.eye(xDim)
                Q0Chol_dxd[:rDim,:rDim] = Q0Chol
                x0 = ( x0scale*np.dot(np.random.randn(self.xDim), Q0Chol_dxd) if 
                       X0data is None else X0data[samp] )
                X_single_samp_1xTxd[0,0] = x0
                
                noise_samps = np.random.randn(NTbins, self.rDim)
                for curr_tbin in range(NTbins-1):
                    curr_X_1x1xd = X_single_samp_1xTxd[:,curr_tbin:curr_tbin+1,:]
                    A_1xrxr = sess.run(A_NTxrxr, feed_dict={feed_key : curr_X_1x1xd})
                    A_rxr = np.squeeze(A_1xrxr, axis=0)
                    X_single_samp_1xTxd[0,curr_tbin+1, :rDim] = ( 
                        np.dot(X_single_samp_1xTxd[0,curr_tbin, :rDim], A_rxr) + 
                        np.dot(noise_samps[curr_tbin+1], QChol) )
                    X_single_samp_1xTxd[0,curr_tbin+1, rDim:] = ( 
                        X_single_samp_1xTxd[0,curr_tbin,rDim:] )
                    
                # Compute MSE and discard path is MSE < path_mse_threshold
                # (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(X_single_samp_1xTxd[0,tbin+1] - 
                                    X_single_samp_1xTxd[0,tbin]) for tbin in 
                                                    range(NTbins-1)])
                samp_norm = Xsamp_mse
        
            Xdata_NxTxd[samp,:,:] = X_single_samp_1xTxd
        
        return Xdata_NxTxd
    

    
    