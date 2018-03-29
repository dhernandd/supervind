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

# Hideous hack to have this code run both as a package and imported from a
# Jupyter notebook. A fairy dies in Neverland every time you run this.s
if __name__ == 'RecognitionModels':
    from LatEvModels import LocallyLinearEvolution #@UnresolvedImport #@UnusedImport
    from utils import blk_tridiag_chol, blk_chol_inv #@UnresolvedImport #@UnusedImport
    from layers import FullLayer #@UnresolvedImport #@UnusedImport
else:
    from .LatEvModels import LocallyLinearEvolution #@Reimport
    from .utils import blk_tridiag_chol, blk_chol_inv #@Reimport
    from .layers import FullLayer #@Reimport

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
            Mu_NxTxd = tf.reshape(Mu_NTxd, [Nsamps, NTbins, xDim], name='MuX')

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
            Lambda_NxTxdxd = tf.reshape(Lambda_NTxdxd, [Nsamps, NTbins, xDim, xDim], name='Lambda')
        
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
        self.TheChol_2xxPxTxdxd, self.postX_PxTxd, self.checks1 = self._compute_TheChol_postX(self.X)
        
        self.Entropy = self.compute_Entropy()

    def _compute_TheChol(self, InputX, InputY=None):
        """
        """
        if InputY: _, Lambda_NxTxdxd, self.LambdaMu_NxTxd = self.get_Mu_Lambda(InputY)
        else: Lambda_NxTxdxd = self.Lambda_NxTxdxd
            
        Nsamps = tf.shape(InputX)[0]
        NTbins = tf.shape(InputX)[1]
        xDim = self.xDim
        
        # WARNING: Some serious tensorflow gymnastics in the next 100 lines or so
        A_NTxdxd = self.lat_ev_model._define_evolution_network(InputX)[0]
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim])
        A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])

        QInv_dxd = self.lat_ev_model.QInv_dxd
        Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
        
        # Constructs the block diagonal matrix:
        #     Qt^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
        QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [Nsamps*(NTbins-1), 1, 1])
        QInvs_Tm2xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [NTbins-2, 1, 1])
        Q0Inv_1xdxd = tf.expand_dims(Q0Inv_dxd, axis=0)
        Q0QInv_Tm1xdxd = tf.concat([Q0Inv_1xdxd, QInvs_Tm2xdxd], axis=0)
        QInvsTot_NTm1xdxd = tf.tile(Q0QInv_Tm1xdxd, [Nsamps, 1, 1])

        # The diagonal blocks of Omega(z) up to T-1:
        #     Omega(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
        AQInvsA_NTm1xdxd = ( tf.matmul(A_NTm1xdxd, tf.matmul(QInvs_NTm1xdxd, A_NTm1xdxd, transpose_b=True)) 
                             + QInvsTot_NTm1xdxd )
        AQInvsA_NxTm1xdxd = tf.reshape(AQInvsA_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])                                     
        
        # The off-diagonal blocks of Omega(z):
        #     Omega(z)_{i,i+1} = -A(z)^T*Q^-1,     for i in {1,..., T-2}
        AQInvs_NTm1xdxd = -tf.matmul(A_NTm1xdxd, QInvs_NTm1xdxd)
#         AQInvs_NTm1xdxd = tf.reshape(AQInvs_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])
        
        # Tile in the last block Omega_TT. 
        # This one does not depend on A. There is no latent evolution beyond T.
        QInvs_Nx1xdxd = tf.tile(tf.reshape(QInv_dxd, shape=[1, 1, xDim, xDim]), [Nsamps, 1, 1, 1])
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
        
        return TheChol_2xNxTxdxd, [A_NTxdxd, AA_NxTxdxd, BB_NxTm1xdxd]
    
    def _compute_postX(self, InputX):
        """
        """
        Nsamps = tf.shape(InputX)[0]
        NTbins = tf.shape(InputX)[1]
        xDim = self.xDim
        
        TheChol_2xNxTxdxd = self.TheChol_2xNxTxdxd
        QInvs_NTm1xdxd = self.QInvs_NTm1xdxd
        A_NTm1xdxd = self.A_NTm1xdxd
        LambdaMu_NxTxd = self.LambdaMu_NxTxd
        
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
#         num_NxTxd = ( LambdaMu_NxTxd + postX_gradterm_NxTxd if self.params.use_grad_term else
#                       LambdaMu_NxTxd )
        postX = tf.scan(fn=aux_fn2, 
                    elems=[TheChol_2xNxTxdxd[0], TheChol_2xNxTxdxd[1],
                           LambdaMu_NxTxd + postX_gradterm_NxTxd],
                    initializer=tf.zeros_like(LambdaMu_NxTxd[0], dtype=DTYPE) )
        postX_ng = tf.scan(fn=aux_fn2, 
                        elems=[TheChol_2xNxTxdxd[0], TheChol_2xNxTxdxd[1], LambdaMu_NxTxd],
                        initializer=tf.zeros_like(LambdaMu_NxTxd[0], dtype=DTYPE) )      
        postX = tf.identity(postX, name='postX')
        postX_ng = tf.identity(postX_ng, name='postX_ng') # tensorflow triple axel! :)
                
        return postX, postX_ng, [postX_gradterm_NxTxd]


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
    
    
    def compute_Entropy(self, Input=None):
        """
        Computes the Entropy. Takes an Input to provide that later on, we can
        add to the graph the Entropy evaluated as a function of the posterior.
        """
        xDim = self.xDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            TheChol_2xxNxTxdxd = self.TheChol_2xxNxTxdxd
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
<<<<<<< HEAD
            TheChol_2xNxTxdxd, _ = self._compute_TheChol(Input) # grads are irrelevant for this
=======
            TheChol_2xxNxTxdxd, _, _ = self._compute_TheChol_postX(Input)
>>>>>>> working all main components for params, missing training
             
        with tf.variable_scope('entropy'):
            self.thechol0 = tf.reshape(TheChol_2xxNxTxdxd[0], 
                                       [Nsamps*NTbins, xDim, xDim])
            LogDet = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(self.thechol0)))
                    
            Nsamps = tf.cast(Nsamps, DTYPE)        
            NTbins = tf.cast(NTbins, DTYPE)        
            xDim = tf.cast(xDim, DTYPE)                
            
            Entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi))*xDim,
                             0.5*LogDet, name='Entropy')  # Yuanjun has xDim here so I put it but I don't think this is right.
        
        return Entropy
    
    
    def get_lat_ev_model(self):
        """
        Auxiliary function for extracting the latent evolution model that the
        Generative Model should use.
        """
        return self.lat_ev_model
    