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
    def __init__(self, X, Ids, params):
        """
        Args:
            X:
            Ids: List of size num_ents of identities for each trial. 
        """
        self.X = X
        self.Ids = Ids
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
            self.QInvChol_dxd = tf.get_variable('QInvChol', initializer=tf.cast(tf.eye(xDim), DTYPE))
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                                  name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True, name='Q')
        
        # Variance of the initial points
        if not hasattr(self, 'Q0InvChol'):
            self.Q0InvChol_dxd = tf.get_variable('Q0InvChol', initializer=tf.cast(tf.eye(xDim), DTYPE))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd, transpose_b=True,
                                   name='Q0Inv')
        
        # The starting coordinates in state-space
        self.x0 = tf.get_variable('x0', initializer=tf.cast(tf.zeros(self.xDim), DTYPE) )
        
        if not hasattr(self, 'alpha'):
            init_alpha = tf.constant_initializer(0.2)
            self.alpha = tf.get_variable('alpha', shape=[], initializer=init_alpha, dtype=DTYPE)

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
        if Ids is None:
            Ids = self.Ids
            if Input is None:
                Input_NxTxd = self.X
            else:
                raise ValueError("You must provide Ids for this Input")
        else:
            Input_NxTxd = Input
        
        Nsamps = tf.shape(Ids)[0]
        NTbins = tf.shape(Input_NxTxd)[1]
        ev_params_Nxp = tf.gather(self.ev_params_Pxp, indices=Ids)
        ev_params_NxTxp = tf.tile(tf.expand_dims(ev_params_Nxp, axis=1), [1, NTbins, 1])
        Input_NxTxr = tf.concat([Input_NxTxd, ev_params_NxTxp], axis=2)

        rangeB = self.params.initrange_B
        evnodes = 64
        Input_NTxr = tf.reshape(Input_NxTxr, [Nsamps*NTbins, rDim])
        fully_connected_layer = FullLayer()
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input_NTxr, evnodes, 'softmax', 'full1')
            output = fully_connected_layer(full1, xDim**2, 'linear', 'output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim], name='B')

        # Broadcast
        A_NTxdxd = self.alpha*B_NTxdxd + self.Alinear_dxd
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
        """
        """
        xDim = self.xDim
        if xin is None: xin = self.x

        singleA_1x1xdxd = self._define_evolution_network(xin, Id=ids)[0]
        singleA_d2 = tf.reshape(singleA_1x1xdxd, [xDim**2])
        grad_list_d2xd = tf.squeeze(tf.stack([tf.gradients(Ai, xin) for Ai
                                              in tf.unstack(singleA_d2)]))

        return grad_list_d2xd
    
    @staticmethod
    def concat_by_id(Input_PxTx, Ids):
        """
        """
        Input_PxxTx = tf.unstack(Input_PxTx)
        return tf.stack([Input_PxxTx[idx] for idx in Ids])
        
    
class LocallyLinearEvolution_wParams(NoisyEvolution_wParams):
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, X, Ids, params):
        """
        """
        NoisyEvolution_wParams.__init__(self, X, Ids, params)
                        
        self.logdensity_Xterms, self.checks_LX = self.compute_LogDensity_Xterms(with_inflow=True)
        
    
    def compute_LogDensity_Xterms(self, Input=None, Ids=None, with_inflow=False):
        """
        """
        xDim = self.xDim
        if Ids is None:
            Ids = self.Ids
            if Input is None:
                X_NxTxd = self.X
                A_NxTxdxd = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
            else:
                raise ValueError("You must provide Ids if you provide an Input")
        else:
            X_NxTxd = self.X if Input is None else Input
            A_NxTxdxd, Awinflow_NxTxdxd, _ = self._define_evolution_network(X_NxTxd, Ids)
            A_NxTxdxd = A_NxTxdxd if not with_inflow else Awinflow_NxTxdxd
        
        NTbins = tf.shape(X_NxTxd)[1]
        Nsamps = tf.shape(Ids)[0]
        Xprime_NxTm1xd = tf.squeeze(tf.matmul(tf.expand_dims(X_NxTxd[:,:-1], axis=2),
                                              A_NxTxdxd[:,:-1]))

        resX_NxTm1x1xd = tf.expand_dims(X_NxTxd[:,1:,:xDim] - Xprime_NxTm1xd, axis=2)
        QInv_NxTm1xdxd = tf.tile(tf.reshape(self.QInv_dxd, [1, 1, xDim, xDim]),
                                 [Nsamps, NTbins-1, 1, 1])
        resX0_Nxd = X_NxTxd[:,0,:xDim] - self.x0
        
        LX1 = -0.5*tf.reduce_sum(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd), name='LX0')
        LX2 = -0.5*tf.reduce_sum(resX_NxTm1x1xd*tf.matmul(resX_NxTm1x1xd, QInv_NxTm1xdxd), name='L2')
        LX3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))
        LX4 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*tf.cast((NTbins-1), DTYPE) )
        LX5 = -0.5*np.log(2*np.pi)*tf.cast(NTbins*xDim, DTYPE)
        
        LatentDensity = LX1 + LX2 + LX3 + LX4 + LX5
        
        return LatentDensity, [LX1, LX2, LX3, LX4, LX5]
    
    
    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, prefix, Ids=None, Nsamps=50, NTbins=30, X0data=None, with_inflow=False,
                 path_mse_threshold=0.7, draw_plots=True):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        
        Get the number of samples. The hierarchy is: i) Get the 0th dimension
        of X0data; ii) if X0data is None, get the 0th dim of Ids; iii) if Ids
        is None, use the provided Nsamps
        """
        print('Sampling...')

        xDim = self.xDim        
        num_ents = self.num_diff_entities
        
        # Get Nsamps
        if X0data is None:
            if Ids is None:
                Ids = np.random.randint(num_ents, size=Nsamps)
            else:
                Ids = np.asarray(Ids)
                Nsamps = len(Ids)
        else:
            Nsamps = X0data.shape[0]
            if Ids is None:
                Ids = np.random.randint(num_ents, size=Nsamps)
            else:
                if len(Ids) != Nsamps:
                    raise ValueError("The length of Ids must equal the length of the initial data, "
                                     "X0data")
                Ids = np.asarray(Ids)

        Xdata_NxTxd = np.zeros([Nsamps, NTbins, xDim]) 
        
        trials = 0
        x0scale = 25.0
        A_NxTxdxd = self.Awinflow_NxTxdxd if with_inflow else self.A_NxTxdxd
        Q0Chol_dxd = sess.run(self.Q0Chol_dxd)
        QChol_dxd = sess.run(self.QChol_dxd)
        for samp in range(Nsamps):
            curr_id = Ids[samp:samp+1]
            samp_norm = 0.0 # needed to avoid paths that start too close to an attractor
            
            # lower path_mse_threshold to keep paths closer to x = const.
            while samp_norm < path_mse_threshold:
                if trials % 10 == 0:
                    print('Trials:', trials)
                X_single_samp_1xTxd = np.zeros([1, NTbins, xDim])
                x0 = ( x0scale*np.dot(np.random.randn(xDim), Q0Chol_dxd) if 
                       X0data is None else X0data[samp] )
                X_single_samp_1xTxd[0,0] = x0
                
                noise_samps = np.random.randn(NTbins, xDim)
                for curr_tbin in range(NTbins-1):
                    curr_X_1x1xd = X_single_samp_1xTxd[:,curr_tbin:curr_tbin+1,:]
#                     A_Txdxd = A_PxxTxdxd[curr_id]
                    A_1x1xdxd = sess.run(A_NxTxdxd, feed_dict={prefix+'X:0' : curr_X_1x1xd,
                                                               prefix+'Ids:0' : curr_id})
                    A_dxd = np.squeeze(A_1x1xdxd, axis=0)
                    X_single_samp_1xTxd[0,curr_tbin+1] = ( 
                        np.dot(X_single_samp_1xTxd[0,curr_tbin], A_dxd) + 
                        np.dot(noise_samps[curr_tbin+1], QChol_dxd) )
                    
                # Compute MSE and discard path if MSE < path_mse_threshold
                # (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(X_single_samp_1xTxd[0,tbin+1] - 
                                    X_single_samp_1xTxd[0,tbin]) for tbin in range(NTbins-1)])
                samp_norm = Xsamp_mse
        
                trials += 1
            Xdata_NxTxd[samp,:,:] = X_single_samp_1xTxd
        
        if draw_plots:
            for ent in range(num_ents):
                print('Plottins DS for entity ', str(ent), '...')
                list_idxs = [i for i, Id in enumerate(Ids) if Id == ent]
                XdataId_NxTxd = Xdata_NxTxd[list_idxs]
                self.plot_2Dquiver_paths(sess, XdataId_NxTxd, [ent], prefix, draw=False,
                                         rslt_file='quiver_plot'+str(ent), with_inflow=with_inflow)
                
        # TODO: Investigate why this is so slow!
        return Xdata_NxTxd, Ids
    
    def eval_nextX(self, session, Xdata, Id, prefix, with_inflow=False):
        """
        Given a symbolic array of points in latent space Xdata = [X0, X1,...,XT], \
        gives the prediction for the next time point
         
        This is useful for the quiver plots.
        
        Args:
            Xdata: Points in latent space at which the dynamics A(X) shall be
            determined.
            with_inflow: Should an inward flow from infinity be superimposed to A(X)?
        """
        NTbins = Xdata.shape[1]
        xDim = self.xDim
        if Xdata.shape[0] != 1:
            raise ValueError("You must pass a single trial (with a specific Id) to eval_nextX")
        Xdata_1xTm1xd = Xdata[:,:-1]
        
#         A_Txdxd = self.A_NxTxdxd[Id] if not with_inflow else self.Awinflow_NxTxdxd[Id]
        A_NxTxdxd = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        A_NTxdxd = tf.reshape(A_NxTxdxd, [-1, xDim, xDim])
        A_Tm1xdxd = session.run(A_NTxdxd, feed_dict={prefix+'X:0' : Xdata_1xTm1xd,
                                                     prefix+'Ids:0' : Id})
        
        Xdata = Xdata[0,:-1,:].reshape(NTbins-1, self.xDim)
        return np.einsum('ij,ijk->ik', Xdata, A_Tm1xdxd).reshape(1, NTbins-1, self.xDim)
    
    @staticmethod
    def define2DLattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(2, -1).T


    def quiver2D_flow(self, session, prefix, Id, clr='black', scale=25,
                      x1range=(-50.0, 50.0), x2range=(-50.0, 50.0), figsize=(13,13), 
                      pause=False, draw=False, with_inflow=True, newfig=True, savefile=None):
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
        
        nextX = self.eval_nextX(session, lattice, Id, prefix, with_inflow=with_inflow)
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

    
    def plot2D_sampleX(self, Xdata, figsize=(13,13), newfig=True, pause=True, draw=True, skipped=1):
        """
        Plots the evolution of the dynamical system in a 2D projection..
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
    
    def plot_2Dquiver_paths(self, session, Xdata, Id, prefix, rlt_dir=TEST_DIR+addDateTime()+'/', 
                            rslt_file='quiver_plot', with_inflow=True, savefig=True, draw=True,
                            pause=False, feed_range=False):
        """
        """
        if savefig:
            if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
            rslt_file = rlt_dir + rslt_file
        
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata, pause=pause, draw=draw, newfig=True)
        if feed_range:
            x1range, x2range = axes.get_xlim(), axes.get_ylim()
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        else:
            x1range = x2range = (-50.0, 50.0)
#             s = 150
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
                    
        self.quiver2D_flow(session, prefix, Id, pause=pause, x1range=x1range, 
                           x2range=x2range, scale=s, newfig=False, 
                           with_inflow=with_inflow, draw=draw)
        if savefig:
            plt.savefig(rslt_file)
        
        plt.close()

    

    
    