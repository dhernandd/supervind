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
import numpy as np

import tensorflow as tf

from .LLinearEv_wParams import LocallyLinearEvolution_wParams
from .utils import blk_tridiag_chol, blk_chol_inv
from .layers import FullLayer

DTYPE = tf.float32

class CellVoltageRecognition():
    """
    """
    def __init__(self, Y, X, params):
        """
        """
        self.params = params

        self.Y = Y
        self.X = X

        self.xDim = params.xDim
        
        self.Mu_1xTxd, self.Lambda_1xTxdxd, self.LambdaMu_1xTxd = self.get_Mu_Lambda(self.Y)
        
        tf.add_to_collection("recog_nns", self.Mu_1xTxd)
        tf.add_to_collection("recog_nns", self.Lambda_1xTxdxd)
        
    def get_Mu_Lambda(self, InputY):
        """
        """
        xDim = self.xDim
        NTbins = tf.shape(InputY)[1]
        
        rangeLambda = self.params.initrange_LambdaX
        rangeX = self.params.initrange_MuX
        rec_nodes = 60
        Y_input_Tx1 = tf.reshape(InputY, [NTbins, 1])
        fully_connected_layer = FullLayer()
        with tf.variable_scope("recog_nn_mu", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Y_input_Tx1, rec_nodes, 'softplus', 'full1',
                                          initializer=tf.random_normal_initializer(stddev=0.1))
            full2 = fully_connected_layer(full1, rec_nodes, 'softplus', 'full2',
                                          initializer=tf.random_normal_initializer(stddev=rangeX))
            Mu_Txdm1 = fully_connected_layer(full2, xDim-1, 'linear', 'output')
            Mu_Txd = tf.concat([tf.identity(Y_input_Tx1), Mu_Txdm1],axis=1)
            Mu_1xTxd = tf.reshape(Mu_Txd, [1, NTbins, xDim])

        with tf.variable_scope("recog_nn_lambda", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Y_input_Tx1, rec_nodes, 'softplus', 'full1',
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            full2 = fully_connected_layer(full1, rec_nodes, 'softplus', 'full2',
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            full3 = fully_connected_layer(full2, xDim**2, 'linear', 'output',
                                        initializer=tf.orthogonal_initializer(gain=rangeLambda))
#                                           initializer=tf.random_normal_initializer(stddev=0.1))
            LambdaChol_Txdxd = tf.reshape(full3, [NTbins, xDim, xDim])
            Lambda_Txdxd = tf.matmul(LambdaChol_Txdxd, LambdaChol_Txdxd,
                                      transpose_b=True)
            Lambda_1xTxdxd = tf.expand_dims(Lambda_Txdxd, axis=0)
        
        LambdaMu_Txd = tf.squeeze(tf.matmul(Lambda_Txdxd, tf.expand_dims(Mu_Txd, axis=2)), axis=2)
        LambdaMu_1xTxd = tf.expand_dims(LambdaMu_Txd, axis=0)
    
        return Mu_1xTxd, Lambda_1xTxdxd, LambdaMu_1xTxd


class SmoothingNLDSTimeSeries(CellVoltageRecognition):
    """
    """
    def __init__(self, Y, X, params):
        """
        """
        CellVoltageRecognition.__init__(self, Y, X, params)
            
        lat_mod_classes = {'llinear' : LocallyLinearEvolution_wParams}
        LatModel = lat_mod_classes[params.lat_mod_class]
        self.lat_ev_model = LatModel(X, params)
                    
        # ***** COMPUTATION OF THE POSTERIOR *****#
        self.TheChol_2xNxTxdxd, self.postX, self.checks1 = self._compute_TheChol_postX(self.X)
         
        self.Entropy = self.compute_Entropy()

    def _compute_TheChol_postX(self, InputX, Id=None, InputY=None):
        """
        Define the evolution map A, Ax_t \sim x_t+1. The behavior of this
        function depends on whether an Id is provided.

        (For the moment implement Id=None and Id=tf.constant()
        """
        if InputY: _, Lambda_1xTxdxd, LambdaMu_1xTxd = self.get_Mu_Lambda(InputY)
        else: Lambda_1xTxdxd, LambdaMu_1xTxd = self.Lambda_1xTxdxd, self.LambdaMu_1xTxd
            
        NTbins = tf.shape(InputX)[1]
        xDim = self.xDim
        Nsamps = self.params.num_diff_entities if Id is None else len(Id)
        
        # WARNING: Some serious tensorflow gymnastics in the next 100 lines or so
        # First define the evolution law. N here can be either P or 1
        A_NxTxdxd = self.lat_ev_model._define_evolution_network(InputX, Id)[0]
        A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])

        # Bring in the evolution variance
        QInv_dxd = self.lat_ev_model.QInv_dxd
        Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
        
        # Constructs the block diagonal matrix:
        #     QQ^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
        QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [Nsamps*(NTbins-1), 1, 1])
        QInvs_NTm2xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [Nsamps*(NTbins-2), 1, 1])
        Q0Inv_1xdxd = tf.expand_dims(Q0Inv_dxd, axis=0)
        Q0QInv_Tm1xdxd = tf.concat([Q0Inv_1xdxd, QInvs_NTm2xdxd], axis=0)
        QInvsTot_NTm1xdxd = tf.tile(Q0QInv_Tm1xdxd, [Nsamps, 1, 1])

        # The diagonal blocks of the full time series variance Omega(Z) up to T-1:
        #     Omega(Z)_ii = A(z_i)^T*QQ_ii^{-1}*A(z_i) + QQ_ii^{-1},     for i in {1,...,T-1 }
        AQInvsA_NTm1xdxd = ( tf.matmul(A_NTm1xdxd, tf.matmul(QInvs_NTm1xdxd, A_NTm1xdxd, transpose_b=True)) 
                            + QInvsTot_NTm1xdxd )
        AQInvsA_NxTm1xdxd = tf.reshape(AQInvsA_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])                                     
        
        # The off-diagonal blocks of Omega(Z):
        #     Omega(Z)_{i,i+1} = -A(z_i)^T*Q_ii^-1,     for i in {1,..., T-2}
        AQInvs_NTm1xdxd = -tf.matmul(A_NTm1xdxd, QInvs_NTm1xdxd)
#         AQInvs_NTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [1, NTbins-1, xDim, xDim])
        
        # Tile in the last block Omega_TT. 
        # This one does not depend on A. There is no latent evolution beyond T.
        QInvs_Nx1xdxd = tf.tile(tf.reshape(QInv_dxd, shape=[1, 1, xDim, xDim]), [Nsamps, 1, 1, 1])
        AQInvsAQInv_NxTxdxd = tf.concat([AQInvsA_NxTm1xdxd, QInvs_Nx1xdxd], axis=1)
        
        # Add in the covariance coming from the observations
        AA_NxTxdxd = Lambda_1xTxdxd + AQInvsAQInv_NxTxdxd
        BB_NxTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])        
        
        # Computation of the Cholesky decomposition for the total covariance
        aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
        TheChol_2xxNxTxdxd = tf.scan(fn=aux_fn1, elems=[AA_NxTxdxd, BB_NxTm1xdxd],
                    initializer=[tf.zeros_like(AA_NxTxdxd[0]), tf.zeros_like(BB_NxTm1xdxd[0])] )
        
        # TODO: Include an option to turn off the computation of gradterm.
        # TODO: Fix the get_grads func to include Ids
        if self.params.use_grads:
            Input_f_Tm1x1xd = tf.reshape(InputX[:,:-1,:], [NTbins-1, 1, xDim])
            Input_b_Tm1x1xd = tf.reshape(InputX[:,1:,:], [NTbins-1, 1, xDim])
            get_grads = lambda xin : self.lat_ev_model.get_A_grads(xin)
            Agrads_Tm1xd2xd = tf.map_fn(get_grads, tf.expand_dims(Input_f_Tm1x1xd, axis=1))
            Agrads_Tm1xdxdxd = tf.reshape(Agrads_Tm1xd2xd, [NTbins-1, xDim, xDim, xDim])
    
            # Move the gradient dimension to the 0 position, then unstack.
            Agrads_split_dxxTm1xdxd = tf.unstack(tf.transpose(Agrads_Tm1xdxdxd, [3, 0, 1, 2]))
    
            # G_k = -0.5(X_i.*A_ij;k.*Q_jl.*A^T_lm.*X_m + X_i.*A_ij.*Q_jl.*A^T_lm;k.*X_m)  
            grad_tt_postX_dxTm1 = -0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(tf.matmul(Input_f_Tm1x1xd, Agrad_Tm1xdxd), 
                    QInvs_NTm1xdxd), A_NTm1xdxd, transpose_b=True),
                    Input_f_Tm1x1xd, transpose_b=True) +
                tf.matmul(tf.matmul(tf.matmul(tf.matmul(
                    Input_f_Tm1x1xd, A_NTm1xdxd), 
                    QInvs_NTm1xdxd), Agrad_Tm1xdxd, transpose_b=True),
                    Input_f_Tm1x1xd, transpose_b=True)
                    for Agrad_Tm1xdxd 
                    in Agrads_split_dxxTm1xdxd]), axis=[2,3] )
            # G_ttp1 = -0.5*X_i*A_ij;k*Q_jl*X_l
            grad_ttp1_postX_dxTm1 = 0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(Input_f_Tm1x1xd, Agrad_Tm1xdxd),
                QInvs_NTm1xdxd), Input_b_Tm1x1xd, transpose_b=True) 
                    for Agrad_Tm1xdxd in Agrads_split_dxxTm1xdxd ]), axis=[2,3])
            # G_ttp1 = -0.5*X_i*Q_ij*A^T_jl;k*X_l
            grad_tp1t_postX_dxTm1 = 0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(Input_b_Tm1x1xd, QInvs_NTm1xdxd),
                Agrad_Tm1xdxd, transpose_b=True), Input_f_Tm1x1xd, transpose_b=True) 
                    for Agrad_Tm1xdxd in Agrads_split_dxxTm1xdxd ]), axis=[2,3])
            gradterm_postX_dxTm1 = ( grad_tt_postX_dxTm1 + grad_ttp1_postX_dxTm1 +
                                  grad_tp1t_postX_dxTm1 )
            
            # The following term is the second term in Eq. (13) in the paper: 
            # https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf 
            zeros_1x1xd = tf.zeros([1, 1, xDim], dtype=DTYPE)
            postX_gradterm_1xTxd = tf.concat([tf.reshape(tf.transpose(gradterm_postX_dxTm1, [1, 0]),
                                                         [1, NTbins-1, xDim]), zeros_1x1xd], axis=1)
        
        def postX_from_chol(tc1, tc2, lm):
            """
            postX_NxTxd = (Lambda1 + S)^{-1}.(Lambda1_ij.*Mu_j + X^T_k.*S_kj;i.*X_j)
            """
            return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), lower=False, transpose=True)
        aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
        
        LambdaMu_NxTxd = tf.tile(LambdaMu_1xTxd, [Nsamps, 1, 1])
        postX_NxTxd = tf.scan(fn=aux_fn2, elems=[TheChol_2xxNxTxdxd[0], TheChol_2xxNxTxdxd[1],
                                                 LambdaMu_NxTxd],
#                                            LambdaMu_1xTxd + postX_gradterm_1xTxd],
                            initializer=tf.zeros_like(LambdaMu_NxTxd[0], dtype=DTYPE),
                            name='postX_NxTxd' )      # tensorflow triple axel! :)
        
        return TheChol_2xxNxTxdxd, postX_NxTxd, [A_NxTxdxd, AA_NxTxdxd, BB_NxTm1xdxd]


    def sample_postX(self):
        """
        """
        Nsamps, NTbins, xDim = self.Nsamps, self.NTbins, self.xDim
        prenoise_NxTxd = tf.random_normal([Nsamps, NTbins, xDim], dtype=DTYPE)
        
        aux_fn = lambda _, seqs : blk_chol_inv(seqs[0], seqs[1], seqs[2],
                                               lower=False, transpose=True)
        noise = tf.scan(fn=aux_fn, elems=[self.TheChol_2xNxTxdxd[0],
                                          self.TheChol_2xNxTxdxd[1], prenoise_NxTxd],
                        initializer=tf.zeros_like(prenoise_NxTxd[0], dtype=DTYPE) )
        noisy_postX = tf.add(self.postX, noise, name='noisy_postX')
                    
        return noisy_postX 
    
    
    def compute_Entropy(self, Input=None, Id=None):
        """
        Computes the Entropy_N. Takes an Input to provide that later on, we can
        add to the graph the Entropy_N evaluated as a function of the posterior.
        """
        xDim = self.xDim
        if Id is None:
            Nsamps = self.params.num_diff_entities
            if Input is None:
                Input = self.X
                TheChol_2xxNxTxdxd = self.TheChol_2xNxTxdxd
            else:
                TheChol_2xxNxTxdxd, _, _ = self._compute_TheChol_postX(Input)
            NTbins = tf.shape(Input)[1]
        else:
            # TODO:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            TheChol_2xxNxTxdxd, _, _ = self._compute_TheChol_postX(Input, Id)
             
        with tf.variable_scope('entropy'):
            self.thechol0_NxTxdxd = TheChol_2xxNxTxdxd[0]
            LogDet_N = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(self.thechol0_NxTxdxd)), axis=1)
                    
            Nsamps = tf.cast(Nsamps, DTYPE)        
            NTbins = tf.cast(NTbins, DTYPE)        
            xDim = tf.cast(xDim, DTYPE)                
            
            Entropy_N = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi))*xDim,
                             0.5*LogDet_N, name='Entropy_N')  # Yuanjun has xDim here so I put it but I don't think this is right.
        
        return Entropy_N
    
    
    def get_lat_ev_model(self):
        """
        Auxiliary function for extracting the latent evolution model that the
        Generative Model should use.
        """
        return self.lat_ev_model