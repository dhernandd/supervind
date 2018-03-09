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

from .LatEvModels import LocallyLinearEvolution, NonLinearEvolution
from .utils import blk_tridiag_chol, blk_chol_inv
from .layers import FullLayer

DTYPE = tf.float32

class GaussianRecognition():
    """
    """
    def __init__(self, Y, X, params):
        """
        """
        self.params = params

        self.Y = Y
        self.X = X

        self.yDim = params.yDim
        self.xDim = params.xDim
                
        self.Nsamps = tf.shape(self.Y)[0]
        self.NTbins = tf.shape(self.Y)[1]
        
        self.Mu_NxTxd, self.Lambda_NxTxdxd, self.LambdaMu_NxTxd = self.get_Mu_Lambda(self.Y)
        tf.add_to_collection("recog_nns", self.Mu_NxTxd)
        tf.add_to_collection("recog_nns", self.Lambda_NxTxdxd)
        
    def get_Mu_Lambda(self, InputY):
        """
        """
        yDim = self.yDim
        xDim = self.xDim
        Nsamps = tf.shape(InputY)[0]
        NTbins = tf.shape(InputY)[1]
        
        rangeLambda = self.params.initrange_LambdaX
        rangeX = self.params.initrange_MuX
        rec_nodes = 60
        Y_input_NTxD = tf.reshape(InputY, [Nsamps*NTbins, yDim])
        fully_connected_layer = FullLayer()
#         batch_norm_layer = BatchNormalizationLayer()
        with tf.variable_scope("recog_nn_mu", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Y_input_NTxD, rec_nodes, 'softplus', 'full1',
                                          initializer=tf.random_normal_initializer(stddev=0.1))
#             bn1 = batch_norm_layer(full1)
            full2 = fully_connected_layer(full1, rec_nodes, 'softplus', 'full2',
                                          initializer=tf.random_normal_initializer(stddev=rangeX))
            Mu_NTxd = fully_connected_layer(full2, xDim, 'linear', 'output')
            Mu_NxTxd = tf.reshape(Mu_NTxd, [Nsamps, NTbins, xDim])

        with tf.variable_scope("recog_nn_lambda", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Y_input_NTxD, rec_nodes, 'softplus', 'full1',
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            full2 = fully_connected_layer(full1, rec_nodes, 'softplus', 'full2',
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            full3 = fully_connected_layer(full2, xDim**2, 'linear', 'output',
                                        initializer=tf.orthogonal_initializer(gain=rangeLambda))
#                                           initializer=tf.random_normal_initializer(stddev=0.1))
            LambdaChol_NTxdxd = tf.reshape(full3, [Nsamps*NTbins, xDim, xDim])
            Lambda_NTxdxd = tf.matmul(LambdaChol_NTxdxd, LambdaChol_NTxdxd,
                                     transpose_b=True)
            Lambda_NxTxdxd = tf.reshape(Lambda_NTxdxd, [Nsamps, NTbins, xDim, xDim])
        
        LambdaMu_NTxd = tf.squeeze(tf.matmul(Lambda_NTxdxd,
                                             tf.expand_dims(Mu_NTxd, axis=2)), axis=2)
        LambdaMu_NxTxd = tf.reshape(LambdaMu_NTxd, [Nsamps, NTbins, xDim])
    
        return Mu_NxTxd, Lambda_NxTxdxd, LambdaMu_NxTxd


class SmoothingNLDSTimeSeries(GaussianRecognition):
    """
    """
    def __init__(self, Y, X, params):
        """
        """
        GaussianRecognition.__init__(self, Y, X, params)
            
        lat_mod_classes = {'llinear' : LocallyLinearEvolution}
        LatModel = lat_mod_classes[params.lat_mod_class]
        self.lat_ev_model = LatModel(X, params)
                    
        # ***** COMPUTATION OF THE POSTERIOR *****#
        self.TheChol_2xNxTxdxd, self.postX, self.checks1 = self._compute_TheChol_postX(self.X)
        
        self.Entropy = self.compute_Entropy()

    def _compute_TheChol_postX(self, InputX, InputY=None):
        """
        """
        if InputY is not None: _, Lambda_NxTxdxd, LambdaMu_NxTxd = self.get_Mu_Lambda(InputY)
        else: Lambda_NxTxdxd, LambdaMu_NxTxd = self.Lambda_NxTxdxd, self.LambdaMu_NxTxd
            
        Nsamps = tf.shape(InputX)[0]
        NTbins = tf.shape(InputX)[1]
        xDim = self.xDim
        
        # WARNING: Some serious tensorflow gymnastics in the next 100 lines or so
        A_NTxdxd = self.lat_ev_model._define_evolution_network(InputX)[0]
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim])

        QInv_dxd = self.lat_ev_model.QInv_dxd
        Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
        A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
        QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
                                   [Nsamps*(NTbins-1), 1, 1])
        
        # Constructs the block diagonal matrix:
        #     Qt^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
        QInvs_Tm2xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
                                   [(NTbins-2), 1, 1])
        Q0Inv_1xdxd = tf.expand_dims(Q0Inv_dxd, axis=0)
        Q0QInv_Tm1xdxd = tf.concat([Q0Inv_1xdxd, QInvs_Tm2xdxd], axis=0)
        QInvsTot_NTm1xdxd = tf.tile(Q0QInv_Tm1xdxd, [Nsamps, 1, 1])

        # The diagonal blocks of Omega(z) up to T-1:
        #     Omega(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
        AQInvsA_NTm1xdxd = ( tf.matmul(A_NTm1xdxd, 
                                       tf.matmul(QInvs_NTm1xdxd, A_NTm1xdxd, transpose_b=True)) 
                                       + QInvsTot_NTm1xdxd )
        AQInvsA_NxTm1xdxd = tf.reshape(AQInvsA_NTm1xdxd,
                                       [Nsamps, NTbins-1, xDim, xDim])                                     
        
        # The off-diagonal blocks of Omega(z):
        #     Omega(z)_{i,i+1} = -A(z)^T*Q^-1,     for i in {1,..., T-2}
        AQInvs_NTm1xdxd = -tf.matmul(A_NTm1xdxd, QInvs_NTm1xdxd)
        AQInvs_NTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])
        
        # Tile in the last block Omega_TT. 
        # This one does not depend on A. There is no latent evolution beyond T.
        QInvs_Nx1xdxd = tf.tile(tf.reshape(QInv_dxd, shape=[1, 1, xDim, xDim]), 
                                [Nsamps, 1, 1, 1])
        AQInvsAQInv_NxTxdxd = tf.concat([AQInvsA_NxTm1xdxd, QInvs_Nx1xdxd], axis=1)
        
        # Add in the covariance coming from the observations
        AA_NxTxdxd = Lambda_NxTxdxd + AQInvsAQInv_NxTxdxd
        BB_NxTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])        
        
        # Computation of the Cholesky decomposition for the total covariance
        aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
        TheChol_2xNxTxdxd = tf.scan(fn=aux_fn1, 
                    elems=[AA_NxTxdxd, BB_NxTm1xdxd],
                    initializer=[tf.zeros_like(AA_NxTxdxd[0]), 
                                 tf.zeros_like(BB_NxTm1xdxd[0])] )
        
        # TODO: Include an option to turn off the computation of gradterm.
        
        Input_f_NTm1x1xd = tf.reshape(InputX[:,:-1,:], [Nsamps*(NTbins-1), 1, xDim])
        Input_b_NTm1x1xd = tf.reshape(InputX[:,1:,:], [Nsamps*(NTbins-1), 1, xDim])
        get_grads = lambda xin : self.lat_ev_model.get_A_grads(xin)
        Agrads_NTm1xd2xd = tf.map_fn(get_grads, 
                                     tf.expand_dims(Input_f_NTm1x1xd, axis=1))
        Agrads_NTm1xdxdxd = tf.reshape(Agrads_NTm1xd2xd,
                                      [Nsamps*(NTbins-1), xDim, xDim, xDim])

        # Move the gradient dimension to the 0 position, then unstack.
        Agrads_split_dxxNTm1xdxd = tf.unstack(tf.transpose(Agrads_NTm1xdxdxd,
                                                         [3, 0, 1, 2]))

        # G_k = -0.5(X_i.*A_ij;k.*Q_jl.*A^T_lm.*X_m + X_i.*A_ij.*Q_jl.*A^T_lm;k.*X_m)  
        grad_tt_postX_dxNTm1 = -0.5*tf.squeeze(tf.stack(
            [tf.matmul(tf.matmul(tf.matmul(tf.matmul(Input_f_NTm1x1xd, Agrad_NTm1xdxd), 
                QInvs_NTm1xdxd), A_NTm1xdxd, transpose_b=True),
                Input_f_NTm1x1xd, transpose_b=True) +
            tf.matmul(tf.matmul(tf.matmul(tf.matmul(
                Input_f_NTm1x1xd, A_NTm1xdxd), 
                QInvs_NTm1xdxd), Agrad_NTm1xdxd, transpose_b=True),
                Input_f_NTm1x1xd, transpose_b=True)
                for Agrad_NTm1xdxd 
                in Agrads_split_dxxNTm1xdxd]), axis=[2,3] )
        # G_ttp1 = -0.5*X_i*A_ij;k*Q_jl*X_l
        grad_ttp1_postX_dxNTm1 = 0.5*tf.squeeze(tf.stack(
            [tf.matmul(tf.matmul(tf.matmul(Input_f_NTm1x1xd, Agrad_NTm1xdxd),
            QInvs_NTm1xdxd), Input_b_NTm1x1xd, transpose_b=True) 
                for Agrad_NTm1xdxd in Agrads_split_dxxNTm1xdxd ]), axis=[2,3])
        # G_ttp1 = -0.5*X_i*Q_ij*A^T_jl;k*X_l
        grad_tp1t_postX_dxNTm1 = 0.5*tf.squeeze(tf.stack(
            [tf.matmul(tf.matmul(tf.matmul(Input_b_NTm1x1xd, QInvs_NTm1xdxd),
            Agrad_NTm1xdxd, transpose_b=True), Input_f_NTm1x1xd, transpose_b=True) 
                for Agrad_NTm1xdxd in Agrads_split_dxxNTm1xdxd ]), axis=[2,3])
        gradterm_postX_dxNTm1 = ( grad_tt_postX_dxNTm1 + grad_ttp1_postX_dxNTm1 +
                              grad_tp1t_postX_dxNTm1 )
        
        # The following term is the second term in Eq. (13) in the paper: 
        # https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf 
        zeros_Nx1xd = tf.zeros([Nsamps, 1, xDim], dtype=DTYPE)
        postX_gradterm_NxTxd = tf.concat(
            [tf.reshape(tf.transpose(gradterm_postX_dxNTm1, [1, 0]),
                       [Nsamps, NTbins-1, xDim]), zeros_Nx1xd], axis=1)
        
        def postX_from_chol(tc1, tc2, lm):
            """
            postX = (Lambda1 + S)^{-1}.(Lambda1_ij.*Mu_j + X^T_k.*S_kj;i.*X_j)
            """
            return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), 
                                lower=False, transpose=True)
        aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
        postX = tf.scan(fn=aux_fn2, 
                    elems=[TheChol_2xNxTxdxd[0], TheChol_2xNxTxdxd[1], 
                            LambdaMu_NxTxd + postX_gradterm_NxTxd],
#                             LambdaMu_NxTxd],
                    initializer=tf.zeros_like(LambdaMu_NxTxd[0], dtype=DTYPE),
                    name='postX' )      # tensorflow triple axel! :)
        
        return TheChol_2xNxTxdxd, postX, [A_NTxdxd, AA_NxTxdxd, BB_NxTm1xdxd, postX_gradterm_NxTxd]


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
    
    
    def compute_Entropy(self, InputX=None):
        """
        Computes the Entropy. Takes an InputX to provide that later on, we can
        add to the graph the Entropy evaluated as a function of the posterior.
        """
        xDim = self.xDim
        if InputX is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            TheChol_2xNxTxdxd = self.TheChol_2xNxTxdxd
        else:
            Nsamps = tf.shape(InputX)[0]
            NTbins = tf.shape(InputX)[1]
            TheChol_2xNxTxdxd, _, _ = self._compute_TheChol_postX(InputX)
             
        with tf.variable_scope('entropy'):
            self.thechol0 = tf.reshape(TheChol_2xNxTxdxd[0], 
                                       [Nsamps*NTbins, xDim, xDim])
            LogDet = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(self.thechol0)))
                    
            Nsamps = tf.cast(Nsamps, DTYPE)        
            NTbins = tf.cast(NTbins, DTYPE)        
            xDim = tf.cast(xDim, DTYPE)                
            
            Entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi))*xDim,
                             0.5*LogDet, name='Entropy') # TODO: Check factor of xDim
        
        return Entropy
    
    def get_lat_ev_model(self):
        """
        Auxiliary function for extracting the latent evolution model that the
        Generative Model should use.
        """
        return self.lat_ev_model
    
    
class SmoothingNLDSTimeSeries2(GaussianRecognition):
    """
    """
    def __init__(self, Y, X, params):
        """
        """
        GaussianRecognition.__init__(self, Y, X, params)
            
        lat_mod_classes = {'nlinear' : NonLinearEvolution}
        LatentModel = (lat_mod_classes[params.lat_mod_class] if hasattr(params, 'lat_mod_class') 
                       else LocallyLinearEvolution)
        self.lat_ev_model = LatentModel(X, params)
                    
        # ***** COMPUTATION OF THE POSTERIOR *****#
        self.postX, self.AA_NxTxdxd, self.BB_NxTm1xdxd = self._compute_postX(self.X)
        
        self.Entropy = self.compute_Entropy()
        
    def _compute_postX(self, InputX, InputY=None):
        """
        """
        if InputY is not None: _, Lambda_NxTxdxd, LambdaMu_NxTxd = self.get_Mu_Lambda(InputY)
        else: Lambda_NxTxdxd, LambdaMu_NxTxd = self.Lambda_NxTxdxd, self.LambdaMu_NxTxd # OK!
            
        Nsamps = tf.shape(InputX)[0]
        NTbins = tf.shape(InputX)[1]
        xDim = self.xDim # OK!
        latm = self.lat_ev_model
        
        QInv_dxd = latm.QInv_dxd
        Q0Inv_dxd = latm.Q0Inv_dxd

        # WARNING: Some serious tensorflow gymnastics in the next 100 lines or so
        nextX_NxTxd = latm.nextX_NxTxd
        X1toT_NxTm1xd = InputX[:,1:,:]
        nextX0toTm1_NxTm1xd = nextX_NxTxd[:,0:-1,:]
        DeltaX_NxTm1xd = X1toT_NxTm1xd - nextX0toTm1_NxTm1xd

        get_grads = lambda xin : latm.get_f_grads(xin)
        InputX_NTx1x1xd = tf.reshape(InputX, [Nsamps*NTbins, 1, 1, xDim])
        nextXgrads_NTxdxd = tf.map_fn(get_grads, InputX_NTx1x1xd) # grad dimension is -1
        nextXgrads_NxTxdxd = tf.reshape(nextXgrads_NTxdxd, [Nsamps, NTbins, xDim, xDim])

        # 0.5*(x1 - f(x0))_i.*Q_ij.*f(x0)_j;k + 0.5*f(x0)_i;k.*Q_ij.*(x1 - f(x0))_j
        #  = (x1 - f(x0))_i.*Q_ij.*f(x0)_j;k
        nextX0grad_Nxdxd = nextXgrads_NxTxdxd[:,0,:,:]
        post_gradterm0_Nx1xd = (tf.matmul(tf.expand_dims(
            tf.matmul(DeltaX_NxTm1xd[:,0,:], QInv_dxd), axis=1), nextX0grad_Nxdxd) )

        #  = (xtp1 - f(xt))_i.*Q_ij.*f(xt)_j;k - (xt - f(xtm1))_i.*Q_ik
        nextXgrad1toTm1_NTm2xdxd = tf.reshape(nextXgrads_NxTxdxd[:,1:-1,:,:],
                                              [Nsamps*(NTbins-2), xDim, xDim])
        post_gradtermk_NxTm2xd = tf.reshape( tf.matmul(tf.expand_dims(
            tf.matmul(tf.reshape(DeltaX_NxTm1xd[:,1:,], [Nsamps*(NTbins-2), xDim]),
                      QInv_dxd), axis=1), nextXgrad1toTm1_NTm2xdxd) + tf.expand_dims(
            tf.matmul(tf.reshape(nextX0toTm1_NxTm1xd[:,:-1,], [Nsamps*(NTbins-2), xDim]),
                      QInv_dxd), axis=1), [Nsamps, NTbins-2, xDim] )
#             tf.matmul(tf.reshape(DeltaX_NxTm1xd[:,:-1,], [Nsamps*(NTbins-2), xDim]),
#                       QInv_dxd), axis=1), [Nsamps, NTbins-2, xDim] )
        
        # f(xtm1))_i.*Q_ik
        post_gradtermT_Nx1xd = tf.expand_dims(tf.matmul(nextX0toTm1_NxTm1xd[:,-1,:],
                                                        QInv_dxd), axis=1)
#         post_gradtermT_Nx1xd = -tf.expand_dims(tf.matmul(DeltaX_NxTm1xd[:,-1,:],
#                                                         QInv_dxd), axis=1)
        
        post_gradterm_NxTxd = tf.concat([post_gradterm0_Nx1xd, post_gradtermk_NxTm2xd,
                                         post_gradtermT_Nx1xd], axis=1)
        post_numerator_NTx1xd = tf.reshape(LambdaMu_NxTxd + post_gradterm_NxTxd,
                                           [Nsamps*NTbins, 1, xDim])
        
        
        Q_Txdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=[0]), [NTbins-1, 1, 1])
        Q0Q_Txdxd = tf.concat([tf.expand_dims(Q0Inv_dxd, axis=[0]), Q_Txdxd], axis=0)
        AA_NTxdxd = tf.reshape(Lambda_NxTxdxd + Q0Q_Txdxd, [Nsamps*NTbins, xDim, xDim])
        AA_NxTxdxd = tf.reshape(AA_NTxdxd, [Nsamps, NTbins, xDim, xDim])
        post_inv_cov_NTxdxd = tf.map_fn(tf.matrix_inverse, AA_NTxdxd)
        postX_NxTxd = tf.reshape(tf.matmul(post_inv_cov_NTxdxd, post_numerator_NTx1xd, transpose_b=True),
                                 [Nsamps, NTbins, xDim]) 
        
        postX_NTx1x1xd = tf.reshape(postX_NxTxd, [Nsamps*NTbins, 1, 1, xDim])
        QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [Nsamps*(NTbins-1), 1, 1])
        gradspostX_NxTxdxd = tf.reshape(tf.map_fn(get_grads, postX_NTx1x1xd),
                                        [Nsamps, NTbins, xDim, xDim]) 
        gradspostX_NTm1xdxd = tf.reshape(gradspostX_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
        BB_NTm1xdxd = -tf.transpose(tf.matmul(QInvs_NTm1xdxd, gradspostX_NTm1xdxd), perm=[0, 2, 1])
        BB_NxTm1xdxd = tf.reshape(BB_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])
            
        # TODO: Add to the entropy the term in the covariance that has only one derivative
        # That did not work assuming no bugs...
        # TODO: Add the terms depending on the jacobian ugh
        
        return postX_NxTxd, AA_NxTxdxd, BB_NxTm1xdxd
    
    def compute_Entropy(self, InputX=None):
        """
        Computes the Entropy. Takes an InputX to provide that later on, we can
        add to the graph the Entropy evaluated as a function of the posterior.
        """
        xDim = self.xDim
        if InputX is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
        else:
            Nsamps = tf.shape(InputX)[0]
            NTbins = tf.shape(InputX)[1]
            
        AA_NxTxdxd = self.AA_NxTxdxd
        BB_NxTm1xdxd = self.BB_NxTm1xdxd
        AA_NTxdxd = tf.reshape(AA_NxTxdxd, [Nsamps*NTbins, xDim, xDim])
#             _, Lambda_NxTxdxd, _ = self.get_Mu_Lambda(InputX)
#             Lambda_NxTxdxd = self.Lambda_NxTxdxd
# 
#             QInv_dxd = self.lat_ev_model.QInv_dxd
#             Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
#             Q_Txdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=[0]), [NTbins-1, 1, 1])
#             Q0Q_Txdxd = tf.concat([tf.expand_dims(Q0Inv_dxd, axis=0), Q_Txdxd], axis=0)
#             
#             AA_NxTxdxd = Lambda_NxTxdxd + Q0Q_Txdxd
#             AA_NTxdxd = tf.reshape(AA_NxTxdxd, [Nsamps*NTbins, xDim, xDim])

        
#         nextXgrads_NTm1xdxd = tf.reshape(self.nextXgrads_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
#         QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
#                                    [Nsamps*(NTbins-1), 1, 1])
#         BB_NTm1xdxd = -tf.transpose(tf.matmul(QInvs_NTm1xdxd, nextXgrads_NTm1xdxd), perm=[0, 2, 1])
#         BB_NxTm1xdxd = tf.reshape(BB_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])      
        
        # Computation of the Cholesky decomposition for the total covariance
        aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
        TheChol_2xNxTxdxd = tf.scan(fn=aux_fn1, 
                    elems=[AA_NxTxdxd, BB_NxTm1xdxd],
                    initializer=[tf.zeros_like(AA_NxTxdxd[0]), 
                                 tf.zeros_like(BB_NxTm1xdxd[0])] )

        Nsamps = tf.cast(Nsamps, DTYPE)
        NTbins = tf.cast(NTbins, DTYPE)
        xDim = tf.cast(xDim, DTYPE)

        LogDet = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(TheChol_2xNxTxdxd[0])))
#         LogDet = -1.0*tf.reduce_sum(tf.log(tf.matrix_determinant(AA_NTxdxd)))
        Entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi))*xDim,
                         0.5*LogDet, name='Entropy') # TODO: Check factor of xDim

        return Entropy

    def get_lat_ev_model(self):
        """
        Auxiliary function for extracting the latent evolution model that the
        Generative Model should use.
        """
        return self.lat_ev_model



        