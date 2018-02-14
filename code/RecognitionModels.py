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
from utils import variable_in_cpu, blk_tridiag_chol, blk_chol_inv


def FullLayer(Input, nodes, input_dim=None, nl='softplus'):
    nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity}
    nonlinearity = nl_dict[nl]
    
    weights_full1 = variable_in_cpu('weights', [input_dim, nodes], 
                          initializer=tf.orthogonal_initializer())
    biases_full1 = variable_in_cpu('biases', [nodes], 
                             initializer=tf.zeros_initializer(dtype=tf.float64))
    full = nonlinearity(tf.matmul(Input, weights_full1) + biases_full1,
                          name='full1')
    return full


class GaussianRecognition():
    """
    """
    def __init__(self,  yDim, xDim, Y, X):
        """
        """
        self.yDim = yDim
        self.xDim = xDim
        self.Y = Y
        self.X = X
        
        self.Nsamps = Nsamps = tf.shape(self.Y)[0]
        self.NTbins = NTbins = tf.shape(self.Y)[1]
        
        rec_nodes = 60
        Y_input_NTxD = tf.reshape(Y, [Nsamps*NTbins, yDim])
        with tf.variable_scope("recog_nn_mu"):
            with tf.variable_scope('full1'):
                full1 = FullLayer(Y_input_NTxD, rec_nodes, yDim, 'softplus')
            with tf.variable_scope('full2'):
                full2 = FullLayer(full1, rec_nodes, rec_nodes, 'softplus')
            with tf.variable_scope('full3'):
                Mu_NTxd = FullLayer(full2, xDim, rec_nodes, 'linear')
            self.Mu_NxTxd = tf.reshape(Mu_NTxd, [Nsamps, NTbins, xDim])
            tf.add_to_collection("recog_nns", self.Mu_NxTxd)

        with tf.variable_scope("recog_nn_lambda"):
            with tf.variable_scope('full1'):
                full1 = FullLayer(Y_input_NTxD, rec_nodes, yDim, 'softplus')
            with tf.variable_scope('full2'):
                full2 = FullLayer(full1, rec_nodes, rec_nodes, 'softplus')
            with tf.variable_scope('full3'):
                full3 = FullLayer(full2, xDim**2, rec_nodes, 'linear')
            LambdaChol_NTxdxd = tf.reshape(full3, [Nsamps*NTbins, xDim, xDim])
            Lambda_NTxdxd = tf.matmul(LambdaChol_NTxdxd, LambdaChol_NTxdxd,
                                     transpose_b=True)
            self.Lambda_NxTxdxd = tf.reshape(Lambda_NTxdxd, [Nsamps, NTbins, xDim, xDim])
            tf.add_to_collection("recog_nns", self.Lambda_NxTxdxd)
        
        LambdaMu_NTxd = tf.squeeze(tf.matmul(Lambda_NTxdxd, 
                                tf.expand_dims(Mu_NTxd, axis=2)), axis=2)
        self.LambdaMu_NxTxd = tf.reshape(LambdaMu_NTxd, [Nsamps, NTbins, xDim])



class SmoothingNLDSTimeSeries(GaussianRecognition):
    """
    """
    def __init__(self, yDim, xDim, Y, X):
        """
        """
        GaussianRecognition.__init__(self, yDim, xDim, Y, X)
            
        # ***** COMPUTATION OF THE POSTERIOR *****#
        self.lat_ev_model = LocallyLinearEvolution(xDim, X)
                    
        self.TheChol_2xNxTxdxd, self.postX = self._compute_TheChol_postX(self.X)
        
        self.Entropy = self.compute_Entropy()

    
    def _compute_TheChol_postX(self, Input):
        """
        """
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]
        xDim = self.xDim
        
        A_NTxdxd, _ = self.lat_ev_model._define_evolution_network(Input)
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim])
        with tf.variable_scope("TheChol"):
            QInv_dxd = self.lat_ev_model.QInv_dxd
            Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
            A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:],
                                         [Nsamps*(NTbins-1), xDim, xDim])
            QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
                                       [Nsamps*(NTbins-1), 1, 1])
            
            # Computes the block diagonal matrix:
            #     Qt^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
            QInvs_Tm2xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
                                       [(NTbins-2), 1, 1])
            Q0Inv_1xdxd = tf.expand_dims(Q0Inv_dxd, axis=0)
            Q0QInv_Tm1xdxd = tf.concat([Q0Inv_1xdxd, QInvs_Tm2xdxd], axis=0)
            QInvsTot_NTm1xdxd = tf.tile(Q0QInv_Tm1xdxd, [Nsamps, 1, 1])
    
            # The diagonal blocks of Omega(z) up to T-1:
            #     Omega(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
            AQInvsA_NTm1xdxd = ( tf.matmul(A_NTm1xdxd, 
                                         tf.matmul(QInvs_NTm1xdxd, A_NTm1xdxd),
                                         transpose_a=True) + QInvsTot_NTm1xdxd )
            AQInvsA_NxTm1xdxd = tf.reshape(AQInvsA_NTm1xdxd,
                                           [Nsamps, NTbins-1, xDim, xDim])                                     
            
            # The off-diagonal blocks of Omega(z):
            #     Omega(z)_{i,i+1} = A(z)^T*Q^-1,     for i in {1,..., T-2} 
            AQInvs_NTm1xdxd = -tf.matmul(A_NTm1xdxd, QInvs_NTm1xdxd,
                                         transpose_a=True)
            AQInvs_NTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])
            
            # Tile in the last block Omega_TT. 
            # This one does not depend on A. There is no latent evolution beyond T.
            QInvs_Nx1xdxd = tf.tile(tf.reshape(QInv_dxd, shape=[1, 1, 2, 2]), 
                                    [Nsamps, 1, 1, 1])
            AQInvsAQInv_NxTxdxd = tf.concat([AQInvsA_NxTm1xdxd, QInvs_Nx1xdxd], axis=1)
            
            # Add in the covariance coming from the observations
            self.AA_NxTxdxd = self.Lambda_NxTxdxd + AQInvsAQInv_NxTxdxd
            self.BB_NxTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])        
            
            # Computation of the Cholesky decomposition for the total covariance
            aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
            TheChol_2xNxTxdxd = tf.scan(fn=aux_fn1, 
                        elems=[self.AA_NxTxdxd, self.BB_NxTm1xdxd],
                        initializer=[tf.zeros_like(self.AA_NxTxdxd[0]), 
                                     tf.zeros_like(self.BB_NxTm1xdxd[0])] )
        
        with tf.variable_scope('postX'):
            # TODO: Include an option to turn off the computation of gradterm.
            
            # Warning: Some serious tensorflow gymnastics below
            Input_f_NTm1x1xd = tf.reshape(Input[:,:-1,:], [Nsamps*(NTbins-1), 1, xDim])
            Input_b_NTm1x1xd = tf.reshape(Input[:,1:,:], [Nsamps*(NTbins-1), 1, xDim])
            get_grads = lambda xin : self.lat_ev_model.get_A_grads(xin)
            Agrads_NTm1xd2xd = tf.map_fn(get_grads, 
                                         tf.expand_dims(Input_f_NTm1x1xd, axis=1))
            Agrads_NTm1xdxdxd = tf.reshape(Agrads_NTm1xd2xd,
                                          [Nsamps*(NTbins-1), xDim, xDim, xDim])

            # Move the gradient dimension to the 0 position, then unstack.
            Agrads_split_dxxNTm1xdxd = tf.unstack(tf.transpose(Agrads_NTm1xdxdxd,
                                                             [3, 0, 1, 2]))

            # G_k = -0.5(X_i*A^T_ij;k*Q_jl*A_lm*X_m + X_i*A^T_ij*Q_jl*A_lm;k*X_m)  
            grad_tt_postX_dxNTm1 = -0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(tf.matmul(
                    Input_f_NTm1x1xd, Agrad_NTm1xdxd, transpose_b=True), 
                    QInvs_NTm1xdxd), A_NTm1xdxd), Input_f_NTm1x1xd, transpose_b=True) +
                tf.matmul(tf.matmul(tf.matmul(tf.matmul(
                    Input_f_NTm1x1xd, A_NTm1xdxd, transpose_b=True), 
                    QInvs_NTm1xdxd), Agrad_NTm1xdxd),
                    Input_f_NTm1x1xd, transpose_b=True)
                    for Agrad_NTm1xdxd 
                    in Agrads_split_dxxNTm1xdxd]), axis=[2,3] )
            # G_ttp1 = -0.5*X_i*A^T_ij;k*Q_jl*X_l
            grad_ttp1_postX_dxNTm1 = 0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(
                Input_f_NTm1x1xd, Agrad_NTm1xdxd, transpose_b=True),
                QInvs_NTm1xdxd), Input_b_NTm1x1xd, transpose_b=True) 
                    for Agrad_NTm1xdxd in Agrads_split_dxxNTm1xdxd ]), axis=[2,3])
            # G_ttp1 = -0.5*X_i*Q_ij*A_jl;k*X_l
            grad_tp1t_postX_dxNTm1 = 0.5*tf.squeeze(tf.stack(
                [tf.matmul(tf.matmul(tf.matmul(
                Input_b_NTm1x1xd, QInvs_NTm1xdxd),
                Agrad_NTm1xdxd), Input_f_NTm1x1xd, transpose_b=True) 
                    for Agrad_NTm1xdxd in Agrads_split_dxxNTm1xdxd ]), axis=[2,3])
            gradterm_postX_dxNTm1 = ( grad_tt_postX_dxNTm1 + grad_ttp1_postX_dxNTm1 +
                                  grad_tp1t_postX_dxNTm1 )
            
            # The following term is the second term in Eq. (13) in the paper: 
            #
            # https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf 
            #
            # Evaluating it confirms that, for this architecture, it is much
            # smaller than the first term. Hence, the Laplace approximation can
            # be safely computed without it. However, if you are using this
            # method, in general keep in mind that the term is there and may be
            # important.
            zeros_Nx1xd = tf.zeros([Nsamps, 1, xDim], dtype=tf.float64)
            self.gradterm_postX_NxTxd = tf.concat(
                [tf.reshape(tf.transpose(gradterm_postX_dxNTm1, [1, 0]),
                           [Nsamps, NTbins-1, xDim]), zeros_Nx1xd], axis=1) # tf triple axel.
            
            def postX_from_chol(tc1, tc2, lm):
                """
                postX = (Lambda1 + Lambda2)^{-1}.Lambda1.Mu
                """
                return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), 
                                    lower=False, transpose=True)
            aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
            postX = tf.scan(fn=aux_fn2, 
                        elems=[TheChol_2xNxTxdxd[0], TheChol_2xNxTxdxd[1], 
                               self.LambdaMu_NxTxd],
                        initializer=tf.zeros_like(self.LambdaMu_NxTxd[0], dtype=tf.float64),
                        name='postX' )
            
        
        return TheChol_2xNxTxdxd, postX


    def sample_postX(self):
        """
        """
        Nsamps, NTbins, xDim = self.Nsamps, self.NTbins, self.xDim
        prenoise_NxTxd = tf.random_normal([Nsamps, NTbins, xDim], dtype=tf.float64)
        
        aux_fn = lambda _, seqs : blk_chol_inv(seqs[0], seqs[1], seqs[2],
                                               lower=False, transpose=True)
        noise = tf.scan(fn=aux_fn, elems=[self.TheChol_2xNxTxdxd[0],
                                          self.TheChol_2xNxTxdxd[1], prenoise_NxTxd],
                        initializer=tf.zeros_like(prenoise_NxTxd[0], dtype=tf.float64) )
        noisy_postX = tf.add(self.postX, noise, name='noisy_postX')
                    
        return noisy_postX 
    
    
    def compute_Entropy(self, Input=None):
        """
        """
        xDim = self.xDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            TheChol_2xNxTxdxd = self.TheChol_2xNxTxdxd
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            TheChol_2xNxTxdxd, _ = self._compute_TheChol_postX(Input)
             
        with tf.variable_scope('entropy'):
            self.thechol0 = tf.reshape(TheChol_2xNxTxdxd[0], 
                                       [Nsamps*NTbins, xDim, xDim])
            LogDet = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(self.thechol0)))
                    
            Nsamps = tf.cast(Nsamps, tf.float64)        
            NTbins = tf.cast(Nsamps, tf.float64)        
            xDim = tf.cast(Nsamps, tf.float64)                
            
            Entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi))*xDim,
                             0.5*LogDet, name='Entropy')  # Yuanjun has xDim here so I put it but I don't think this is right.
        
        return Entropy
    
    
    def get_lat_ev_model(self):
        """
        Auxiliary function for extracting the latent evolution model that the
        Generative Model should use.
        """
        return self.lat_ev_model
    