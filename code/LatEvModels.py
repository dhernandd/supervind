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

import os

import numpy as np

import tensorflow as tf

from datetools import addDateTime
from utils import variable_in_cpu

TEST_DIR = '/Users/danielhernandez/work/supervind/tests/test_results/'

def flow_modulator(x, x0=30.0, a=0.08):
    return (1 - np.tanh(a*(x - x0)))/2

def flow_modulator_tf(X, x0=30.0, a=0.08):
    return tf.cast((1.0 - tf.tanh(a*(X - x0)) )/2, tf.float64)



class LocallyLinearEvolution():
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, xDim, X):
        """
        Args:
            LatPars:
            xDim:
            X:
            nnname:
        """        
        self.xDim = xDim
        self.X = X
        
        self.Nsamps = Nsamps = tf.shape(self.X)[0]
        self.NTbins = NTbins = tf.shape(self.X)[1]
         
        if not hasattr(self, 'QInvChol'):
            self.QInvChol_dxd = tf.get_variable('QInvChol', 
                                initializer=tf.cast(tf.eye(xDim), tf.float64))
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                              name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True,
                           name='Q')
         
        if not hasattr(self, 'Q0InvChol'):
            self.Q0InvChol_dxd = tf.get_variable('Q0InvChol', 
                                initializer=tf.cast(tf.eye(xDim), tf.float64))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd,
                                   transpose_b=True, name='Q0Inv')
         
        self.x0 = tf.get_variable('x0', 
                                  initializer=tf.cast(tf.zeros(self.xDim), tf.float64) )
         
        if not hasattr(self, 'alpha'):
            init_alpha = tf.constant_initializer(0.2)
            self.alpha = alpha = tf.get_variable('alpha', shape=[], 
                                                 initializer=init_alpha,
                                                 dtype=tf.float64)
#       Define base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear_dxd = tf.get_variable('Alinear', initializer=np.eye(xDim),
                                               dtype=tf.float64)

        evnodes = 64
        X_input = tf.reshape(self.X, [Nsamps*NTbins, xDim], name='X_input')
        with tf.variable_scope("Evolution_Network"):
            with tf.variable_scope('full1'):
                weights_full1 = variable_in_cpu('weights', [xDim, evnodes], 
                                      initializer=tf.random_normal_initializer())
                biases_full1 = variable_in_cpu('biases', [evnodes], 
                                         initializer=tf.constant_initializer())
                full1 = tf.nn.softmax(tf.matmul(X_input, weights_full1) + biases_full1,
                                      name='full1')
            with tf.variable_scope('full2'):
                weights_full2 = variable_in_cpu('weights', [evnodes, xDim**2], 
                                      initializer=tf.random_normal_initializer())
                biases_full2 = variable_in_cpu('biases', [xDim**2], 
                                         initializer=tf.constant_initializer())
                full2 = tf.add(tf.matmul(full1, weights_full2), biases_full2,
                                      name='full2')
            self.B_NTxdxd = tf.reshape(full2, [Nsamps*NTbins, xDim, xDim], name='B')
        
        # Broadcast
        self.A_NTxdxd = alpha*self.B_NTxdxd + self.Alinear_dxd
        
        X_norms = tf.norm(X_input, axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        eye_swap = tf.cast(tf.transpose(tf.tile(tf.expand_dims(tf.eye(self.xDim), 0), 
                                        [Nsamps*NTbins, 1, 1]), [2,1,0]), tf.float64)
        self.Awinflow_NTxdxd = tf.transpose(fl_mod*tf.transpose(
            self.A_NTxdxd, [2,1,0]) + 0.9*(1.0-fl_mod)*eye_swap, [2,1,0])
        
        self.totalB_NxTxdxd = tf.reshape(self.B_NTxdxd, 
                                           [Nsamps, NTbins, xDim, xDim])
        self.totalA_NxTxdxd = tf.reshape(self.A_NTxdxd, 
                                           [Nsamps, NTbins, xDim, xDim])
#
        self.totalA_winflow_NxTxdxd = tf.reshape(self.Awinflow_NTxdxd, 
                                           [Nsamps, NTbins, xDim, xDim])
#
#         
#         # Compute the gradients of B.
#         # TODO: Comment all this so that you know what you are doing.
#         # TODO: Actually implement this thing. Major modifications in the code required.
#         def _compute_B_grads(self):
#             flatBs_NTm1dd = Bs.flatten()
#             
#             # The problem is that this function computes the gradient w.r.t. all
#             # the elements in self.X. This is absolutely unnecessary - the
#             # entries in B only depend on one particular X_t - and
#             # unfortunately, it turns a computation that should be O(N) in one
#             # that is O(N^2*T), obviously increasing the running time like
#             # hell. So it seems I will have to do all this gradient stuff old
#             # school.
#             grad_flatBs_NTm1ddxNxTxd, _ = theano.scan(lambda i, b, x: T.grad(b[i], x), 
#                                                 sequences=T.arange(flatBs_NTm1dd.shape[0]),
#                                                 non_sequences=[flatBs_NTm1dd, self.X])
#             grad_flatBs_NTm1ddxNxTm1xd = grad_flatBs_NTm1ddxNxTxd[:,:,:-1,:]
#             grad_flatBs_NTm1xddxNTm1xd = grad_flatBs_NTm1ddxNxTm1xd.reshape([Nsamps*(Tbins-1),
#                                                                 xDim**2, Nsamps*(Tbins-1), xDim])
#             grad_flatBs_NTm1xdxdxd, _ = theano.scan(lambda i, X: X[i,:,i,:].reshape([xDim, xDim, xDim]).dimshuffle(2, 0, 1), 
#                             sequences=T.arange(Nsamps*(Tbins-1)),
#                             non_sequences=[grad_flatBs_NTm1xddxNTm1xd])
#             self.totalgradBs = grad_flatBs_NTm1xdxdxd.reshape([Nsamps, Tbins-1, xDim, xDim, xDim])
#     #         This works.
#     #         sampleX = np.random.rand(80, 30, 2)
#     #         print 'Finished gradients'
#     #         print 'GradBB:', self.totalgradBs.eval({self.X : sampleX})
#     #         T1 = self.alpha*T.dot(self.totalgradBs.dimshuffle(0,1,2,4,3), T.dot(self.QInv, self.Alinear))
#     #         T1_func = theano.function([self.X], T1)
#     #         print 'T1:', T1_func(sampleX)
#     #         sampleX2 = np.random.rand(80, 30, 2)
#     #         print 'T1a:', T1_func(sampleX2)
#         
# 

    def compute_LogDensity_Xterms(self):
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
        Nsamps = self.Nsamps
        NTbins = self.NTbins
        xDim = self.xDim
        
        totalA_NTm1xdxd = tf.reshape(self.totalA_NxTxdxd[:,:-1,:,:], 
                                     [Nsamps*(NTbins-1), xDim, xDim])
        Xin_NTm1x1xd = tf.reshape(self.X[:,:-1,:], [Nsamps*(NTbins-1), 1, xDim])
        Xprime_NTm1xd = tf.reshape(tf.matmul(Xin_NTm1x1xd, totalA_NTm1xdxd), 
                                    [Nsamps*(NTbins-1), xDim])

        resX_NTm1xd = ( tf.reshape(self.X[:,1:,:], [Nsamps*(NTbins-1), xDim])
                                    - Xprime_NTm1xd )
        resX0_Nxd = self.X[:,0,:] - self.x0
        
        # L = -0.5*(∆X_0^T·Q0^{-1}·∆X_0) - 0.5*Tr[∆X^T·Q^{-1}·∆X] + 0.5*N*log(Det[Q0^{-1}])
        #     + 0.5*N*T*log(Det[Q^{-1}]) - 0.5*N*T*d_X*log(2*Pi)
        L1 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX0_Nxd, self.Q0Inv_dxd), 
                             resX0_Nxd, transpose_b=True), name='L1') 
        L2 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX_NTm1xd, self.QInv_dxd), 
                             resX_NTm1xd, transpose_b=True), name='L2' )
        L3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))*tf.cast(Nsamps, tf.float64)
        L4 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*
               tf.cast((NTbins-1)*Nsamps, tf.float64) )
        L5 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*xDim, tf.float64)
        
#         L1 = -0.5*(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd)).sum()
#         L2 = -0.5*(resX_NxTm1xd*T.dot(resX_NxTm1xd, self.QInv)).sum()
#         L3 = 0.5*T.log(Tnla.det(self.Q0Inv))*Nsamps
#         L4 = 0.5*T.log(Tnla.det(self.QInv))*(Tbins-1)*Nsamps
#         L5 = -0.5*(self.xDim)*np.log(2*np.pi)*Nsamps*Tbins
        LatentDensity = L1 + L2 + L3 + L4 + L5
                
        return LatentDensity, [L1, L2, L3, L4, L5]
    

    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used as standalone

    def sample_X(self, sess, Nsamps=50, NTbins=100, X0data=None, inflow_scale=0.9, 
                 with_inflow=False, path_mse_threshold=1.0, init_from_save=False,
                 draw_plots=False, init_variables=True):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
#         sess = tf.Session() if session is None else session
#         with sess:
        if init_variables: sess.run(tf.global_variables_initializer())
        
        Q0Chol = sess.run(self.Q0Chol_dxd)
        QChol = sess.run(self.QChol_dxd)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, self.xDim])
        x0scale = 25.0
        
        for samp in range(Nsamps):
            # needed to avoid paths that start too close to an attractor
            samp_norm = 0.0
            
            # lower path_mse_threshold to keep paths closer to trivial
            # trajectories, x = const.
            while samp_norm < path_mse_threshold:
                X_single_samp_1xTxd = np.zeros([1, NTbins, self.xDim])
                x0 = ( x0scale*np.dot(np.random.randn(self.xDim), Q0Chol) if 
                       X0data is None else X0data[samp] )
                X_single_samp_1xTxd[0,0] = x0
            
                noise_samps = np.random.randn(NTbins, self.xDim)                                        
                for curr_tbin in range(NTbins-1):
                    curr_X_1x1xd = X_single_samp_1xTxd[:,curr_tbin:curr_tbin+1,:]
                    A_1xdxd = sess.run(self.A_NTxdxd, 
                                       feed_dict = {'X:0' : curr_X_1x1xd})
                    if with_inflow:
                        curr_X_norm = np.linalg.norm(np.squeeze(curr_X_1x1xd, 1))
                        flow_mod = flow_modulator(curr_X_norm)
                        id_likeA = np.expand_dims(np.eye(self.xDim), 0)
                        A_1xdxd = ( flow_mod*A_1xdxd + 
                                    inflow_scale*(1.0-flow_mod)*id_likeA )
                    A_dxd = np.squeeze(A_1xdxd, axis=0)

                    X_single_samp_1xTxd[0,curr_tbin+1,:] = ( 
                        np.dot(X_single_samp_1xTxd[0,curr_tbin,:], A_dxd) + 
                        np.dot(noise_samps[curr_tbin+1], QChol) )
            
                # Compute MSE and discard path is MSE < path_mse_threshold
                # (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(X_single_samp_1xTxd[0,tbin+1] - 
                                    X_single_samp_1xTxd[0,tbin]) for tbin in 
                                                    range(NTbins-1)])
                samp_norm = Xsamp_mse
        
            Xdata_NxTxd[samp,:,:] = X_single_samp_1xTxd
        
        if draw_plots:
            self.plot_2Dquiver_paths(Xdata_NxTxd, sess, with_inflow=with_inflow)

        return Xdata_NxTxd        
    
    
    def eval_nextX(self, Xdata, session, with_inflow=False):
        """
        Given a symbolic array of points in latent space Xdata = [X0, X1,...,XT], \
        gives the prediction for the next time point
         
        This is useful for the quiver plots.
        
        Args:
            Xdata: Points in latent space at which the dynamics A(X) shall be
            determined.
            with_inflow: Should an inward flow from infinity be superimposed to A(X)?
        """
        Nsamps, Tbins = Xdata.shape[0], Xdata.shape[1]
        
        totalA = ( self.totalA_NxTxdxd if not with_inflow 
                   else self.totalA_winflow_NxTxdxd )
        A = session.run(totalA, feed_dict={'X:0' : Xdata})
        A = A[:,:-1,:,:].reshape(Nsamps*(Tbins-1), self.xDim, self.xDim)
        Xdata = Xdata[:,:-1,:].reshape(Nsamps*(Tbins-1), self.xDim)
                
        return np.einsum('ij,ijk->ik', Xdata, A).reshape(Nsamps, Tbins-1, self.xDim)
    
    @staticmethod
    def define2DLattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(2, -1).T


    def quiver2D_flow(self, session, clr='black', scale=50,
                      x1range=(-35.0, 35.0), x2range=(-35.0, 35.0), figsize=(13,13), 
                      pause=True, draw=True, with_inflow=False, newfig=True, savefile=None):
        """
        TODO: Write the docstring for this bad boy.
        """
        import matplotlib.pyplot as plt
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        lattice = self.define2DLattice(x1range, x2range)
        Tbins = lattice.shape[0]
        lattice = np.reshape(lattice, [1, Tbins, self.xDim])
        
        nextX = self.eval_nextX(lattice, session, with_inflow=with_inflow)
        nextX = nextX.reshape(Tbins-1, self.xDim)
        X = lattice[:,:-1,:].reshape(Tbins-1, self.xDim)

        plt.quiver(X.T[0], X.T[1], nextX.T[0]-X.T[0], nextX.T[1]-X.T[1], 
                   color=clr, scale=scale)
        axes = plt.gca()
        axes.set_xlim(x1range)
        axes.set_ylim(x2range)
        if draw: plt.draw()  
        
        if pause:
            plt.pause(0.001)
            input('Press Enter to continue.')
        
        if savefile is not None:
            plt.savefig(savefile)
        else:
            pass
    
    def plot2D_sampleX(self, Xdata, figsize=(13,13), newfig=True, 
                       pause=True, draw=True, skipped=1):
        """
        Plots the evolution of the dynamical system in a 2D projection.
         
        """
        import matplotlib.pyplot as plt
        
        ctr = 0
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        for samp in Xdata:
            if ctr % skipped == 0:
                plt.plot(samp[:,0], samp[:,1], linewidth=2)
                plt.plot(samp[0,0], samp[0,1], 'bo')
                axes = plt.gca()
            ctr += 1
        if draw: plt.draw()  
        if pause:
            plt.pause(0.001)
            input('Press Enter to continue.')
            
        return axes
    
    
    def plot_2Dquiver_paths(self, Xdata, session, rlt_dir=TEST_DIR+addDateTime()+'/', 
                            rslt_file='test_plot', with_inflow=False):
        """
        """
        if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
        rslt_file = rlt_dir + rslt_file
        
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata, pause=False, draw=False, newfig=True)
        x1range, x2range = axes.get_xlim(), axes.get_ylim()
        s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        
        self.quiver2D_flow(session, pause=False, x1range=x1range, 
                           x2range=x2range, scale=s, newfig=False, 
                           with_inflow=with_inflow)
        plt.savefig(rslt_file)
        plt.close()
        


    