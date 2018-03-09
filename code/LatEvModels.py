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

import numpy as np

import tensorflow as tf

# Hideous hack to have this code run both as a package and imported from a
# Jupyter notebook. A fairy dies in Neverland every time you run this.s
if __name__ == 'LatEvModels':
    from datetools import addDateTime #@UnresolvedImport #@UnusedImport
    from layers import FullLayer #@UnresolvedImport #@UnusedImport
else:
    from .datetools import addDateTime #@ @Reimport
    from .layers import FullLayer  # @Reimport


TEST_DIR = './tests/test_results/'
DTYPE = tf.float32

def flow_modulator(x, x0=30.0, a=0.08):
    return (1 - np.tanh(a*(x - x0)))/2

def flow_modulator_tf(X, x0=30.0, a=0.08):
    return tf.cast((1.0 - tf.tanh(a*(X - x0)) )/2, DTYPE)


class NoisyEvolution():
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
            xDim:
            X:
            nnname:
        """        
        self.X = X
        self.params = params

        self.xDim = xDim = params.xDim
        self.init_Q0 = init_Q0 = params.init_Q0
        self.init_Q = init_Q = params.init_Q
        self.alpha = alpha = params.alpha
        
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]
        
        # Variance (Q) of the state-space evolution. 
        if not hasattr(self, 'QInvChol'):
            self.QInvChol_dxd = tf.get_variable('QInvChol', 
                                initializer=tf.cast(init_Q*tf.eye(xDim), DTYPE))
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                              name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True,
                           name='Q')
        
        # Variance of the initial points
        if not hasattr(self, 'Q0InvChol'):
            self.Q0InvChol_dxd = tf.get_variable('Q0InvChol',
                                                 initializer=tf.cast(init_Q0*tf.eye(xDim), DTYPE))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd,
                                   transpose_b=True, name='Q0Inv')
        
        # The starting coordinates in state-space
        self.x0 = tf.get_variable('x0', initializer=tf.cast(tf.zeros(self.xDim), DTYPE))
         
        if not hasattr(self, 'alpha'):
            init_alpha = tf.constant_initializer(alpha)
            self.alpha = tf.get_variable('alpha', shape=[], initializer=init_alpha,
                                         dtype=DTYPE, trainable=False)

    @staticmethod
    def define2DLattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(2, -1).T

    def eval_nextX(self):
        """
        """
        raise NotImplementedError("This is a placeholder method. Please define me")
    
    def quiver2D_flow(self, session, Xvar_name, clr='black', scale=25,
                      x1range=(-35.0, 35.0), x2range=(-35.0, 35.0), figsize=(13,13), 
                      pause=False, draw=False, with_inflow=False, newfig=True, savefile=None):
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
        
        nextX = self.eval_nextX(session, lattice, Xvar_name, with_inflow=with_inflow)
        nextX = nextX.reshape(Tbins, self.xDim)
        X = lattice.reshape(Tbins, self.xDim)

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
    
    def plot_2Dquiver_paths(self, session, Xdata, Xvar_name, rlt_dir=TEST_DIR+addDateTime()+'/', 
                            rslt_file='quiver_plot', with_inflow=False, savefig=False, draw=False,
                            pause=False):
        """
        """
        if savefig:
            if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
            rslt_file = rlt_dir + rslt_file
        
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata, pause=pause, draw=draw, newfig=True)
        x1range, x2range = axes.get_xlim(), axes.get_ylim()
        s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        
        self.quiver2D_flow(session, Xvar_name, pause=pause, x1range=x1range, 
                           x2range=x2range, scale=s, newfig=False, 
                           with_inflow=with_inflow, draw=draw)
        if savefig:
            plt.savefig(rslt_file)
        else:
            pass
        plt.close()

class LocallyLinear(NoisyEvolution):
    """
    """
    def __init__(self, X, params):
        """
        """
        NoisyEvolution.__init__(self, X, params)
#       Define base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear_dxd = tf.get_variable('Alinear', initializer=tf.eye(self.xDim),
                                               dtype=DTYPE)
        
        # Define the evolution for *this* instance
        self.A_NxTxdxd, self.Awinflow_NxTxdxd, self.B_NxTxdxd = self._define_evolution_network()

    def get_A_grads(self, xin=None):
        xDim = self.xDim
        if xin is None: xin = self.x

        singleA_1x1xdxd = self._define_evolution_network(xin)[0]
        singleA_d2 = tf.reshape(singleA_1x1xdxd, [xDim**2])
        grad_list_d2xd = tf.squeeze(tf.stack([tf.gradients(Ai, xin) for Ai
                                              in tf.unstack(singleA_d2)]))

        return grad_list_d2xd 
    
    def _define_evolution_network(self, Input=None):
        """
        """
        xDim = self.xDim
        
        if Input is None:
            Input = self.X
            Nsamps = self.Nsamps
            NTbins = self.NTbins
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
        
        alpha = self.alpha
        rangeB = self.params.initrange_B
        evnodes = 200
        Input = tf.reshape(Input, [Nsamps*NTbins, xDim])
        fully_connected_layer = FullLayer(collections=['EVOLUTION_PARS'])
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input, evnodes, 'softmax', 'full1')
#             full2 = fully_connected_layer(full1, evnodes//2, 'relu', 'full2',
#                                           initializer=tf.orthogonal_initializer())
            output = fully_connected_layer(full1, xDim**2, nl='linear', scope='output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim])
        
        A_NTxdxd = alpha*B_NTxdxd + self.Alinear_dxd # Broadcast
        
        X_norms = tf.norm(Input, axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        eye_swap = tf.transpose(tf.tile(tf.expand_dims(tf.eye(self.xDim), 0),
                                        [Nsamps*NTbins, 1, 1]), [2,1,0])
        Awinflow_NTxdxd = tf.transpose(fl_mod*tf.transpose(
            A_NTxdxd, [2,1,0]) + 0.9*(1.0-fl_mod)*eye_swap, [2,1,0])
        
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim], name='A')
        Awinflow_NxTxdxd = tf.reshape(Awinflow_NTxdxd, 
                                      [Nsamps, NTbins, xDim, xDim], name='Awinflow')
        
        return A_NxTxdxd, Awinflow_NxTxdxd, B_NxTxdxd


class LocallyLinearEvolution(LocallyLinear):
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
        """
        LocallyLinear.__init__(self, X, params)
                        
        self.logdensity_Xterms = self.compute_LogDensity_Xterms()
        
    def compute_LogDensity_Xterms(self, Input=None, with_inflow=False):
        """
        Computes the symbolic log p(X, Y).         
        Inputs:
        """
        xDim = self.xDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            totalA_NxTxdxd = ( self.A_NxTxdxd if not with_inflow else 
                               self.Awinflow_NxTxdxd )
            X = self.X
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            AsBs = self._define_evolution_network(Input)
            A_NxTxdxd, Awinflow_NTxdxd = AsBs[0], AsBs[1]
            totalA_NxTxdxd = A_NxTxdxd if not with_inflow else Awinflow_NTxdxd
            X = Input

        totalA_NTm1xdxd = tf.reshape(totalA_NxTxdxd[:,:-1,:,:], 
                                     [Nsamps*(NTbins-1), xDim, xDim])
        Xin_NTm1x1xd = tf.reshape(X[:,:-1,:], [Nsamps*(NTbins-1), 1, xDim])
        Xprime_NTm1xd = tf.reshape(tf.matmul(Xin_NTm1x1xd, totalA_NTm1xdxd), 
                                    [Nsamps*(NTbins-1), xDim])

        resX_NTm1xd = ( tf.reshape(X[:,1:,:], [Nsamps*(NTbins-1), xDim])
                                    - Xprime_NTm1xd )
        resX0_Nxd = X[:,0,:] - self.x0
        
        # L = -0.5*(resX_0^T*Q0^{-1}*resX_0) - 0.5*Tr[resX^T*Q^{-1}*resX] + 0.5*N*log(Det[Q0^{-1}])
        #     + 0.5*N*T*log(Det[Q^{-1}]) - 0.5*N*T*d_X*log(2*Pi)
        LX0 = -0.5*tf.reduce_sum(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd), name='LX0') 
        LX1 = -0.5*tf.reduce_sum(resX_NTm1xd*tf.matmul(resX_NTm1xd, self.QInv_dxd), name='LX1' )
        LX2 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))*tf.cast(Nsamps, DTYPE)
        LX3 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*
               tf.cast((NTbins-1)*Nsamps, DTYPE) )
        LX4 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*xDim, DTYPE)
        
        LatentDensity = LX0 + LX1 + LX2 + LX3 + LX4
        
        self.LX_summ = tf.summary.scalar('LogDensity_Xterms', LatentDensity)
        self.LX1_summ = tf.summary.scalar('LX1', LX1)
                
        return LatentDensity, [LX0, LX1, LX2, LX3, LX4] 
    
    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, Xvar_name, Nsamps=2, NTbins=3, X0data=None, with_inflow=False,
                 path_mse_threshold=0.1, draw_plots=False):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        print('Sampling from latent dynamics...')        
        xDim = self.xDim
        Q0Chol = sess.run(self.Q0Chol_dxd)
        QChol = sess.run(self.QChol_dxd)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, self.xDim])
        x0scale = 15.0
        
        A_NxTxdxd = self.Awinflow_NxTxdxd if with_inflow else self.A_NxTxdxd
        A_NTxdxd = tf.reshape(A_NxTxdxd, shape=[-1, xDim, xDim])
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
                    A_1xdxd = sess.run(A_NTxdxd, feed_dict={Xvar_name : curr_X_1x1xd})
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
            self.plot_2Dquiver_paths(sess, Xdata_NxTxd, Xvar_name, with_inflow=with_inflow)

        return Xdata_NxTxd        
    
    
    def eval_nextX(self, session, Xdata, Xvar_name, with_inflow=False):
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
        
        totalA = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        A = session.run(totalA, feed_dict={Xvar_name : Xdata})
        A = A.reshape(Nsamps*Tbins, self.xDim, self.xDim)
        Xdata = Xdata.reshape(Nsamps*Tbins, self.xDim)
                
        return np.einsum('ij,ijk->ik', Xdata, A).reshape(Nsamps, Tbins-1, self.xDim)


class LocallyLinearEvolution_wInput(LocallyLinearEvolution):
    """
    TODO: Finish!
    """
    def __init__(self, xDim, X, iDim, ext_input, ext_input_scale):
        """
        """
        NoisyEvolution.__init__(self, xDim, X)
        
        self.I = ext_input
        self.Iscale = ext_input_scale
        self.iDim = iDim
        
        self.X_i_NxTxd = self.input_to_latent()

        self.logdensity_Xterms = self.compute_LogDensity_Xterms(with_inflow=True)
        
    def input_to_latent(self, IInput=None):
        """
        """
        Nsamps = self.Nsamps
        NTbins = self.NTbins
        iDim = self.iDim
        xDim = self.xDim
        if IInput is None: IInput = self.I 
        
        IInput = tf.reshape(IInput, [Nsamps*NTbins], iDim)
        in_nodes = 64
        fully_connected_layer = FullLayer()
        with tf.variable_scope("input_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(IInput, in_nodes, iDim, 'softplus', 'full1')
            full2 = fully_connected_layer(full1, in_nodes, in_nodes, 'softplus', 'full2')
            full3 = fully_connected_layer(full2, xDim, in_nodes, 'linear')
        
        return tf.reshape(full3, [Nsamps, NTbins, xDim])
        
    def compute_LogDensity_Xterms(self, Input=None, IInput=None, with_inflow=False):
        """
        """
        xDim = self.xDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            totalA_NxTxdxd = ( self.A_NxTxdxd if not with_inflow else 
                               self.Awinflow_NxTxdxd )
            X = self.X
            X_i = self.X_i_NxTxd
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            A_NxTxdxd, Awinflow_NTxdxd = self._define_evolution_network(Input)
            totalA_NxTxdxd = A_NxTxdxd if not with_inflow else Awinflow_NTxdxd
            X = Input
            X_i = self.input_to_latent(IInput)
            
        totalA_NTm1xdxd = tf.reshape(totalA_NxTxdxd[:,:-1,:,:], 
                                     [Nsamps*(NTbins-1), xDim, xDim])
        Xin_NTm1x1xd = tf.reshape(X[:,:-1,:], [Nsamps*(NTbins-1), 1, xDim])
        Xprime_NTm1xd = tf.reshape(tf.matmul(Xin_NTm1x1xd, totalA_NTm1xdxd), 
                                    [Nsamps*(NTbins-1), xDim])
        X_i_NTm1xd = tf.reshape(X_i[:,1:,:], [Nsamps*(NTbins-1), xDim])

        resX_NTm1xd = ( tf.reshape(X[:,1:,:], [Nsamps*(NTbins-1), xDim])
                                    - Xprime_NTm1xd - X_i_NTm1xd)

        resX0_Nxd = X[:,0,:] - self.x0 - X_i[:,0,:]

        L1 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX0_Nxd, self.Q0Inv_dxd), 
                             resX0_Nxd, transpose_b=True), name='L1') 
        L2 = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(resX_NTm1xd, self.QInv_dxd), 
                             resX_NTm1xd, transpose_b=True), name='L2' )
        L3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))*tf.cast(Nsamps, DTYPE)
        L4 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*
               tf.cast((NTbins-1)*Nsamps, DTYPE) )
        L5 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*xDim, DTYPE)

        LatentDensity = L1 + L2 + L3 + L4 + L5
                
        return LatentDensity, [L1, L2, resX_NTm1xd, Xin_NTm1x1xd, totalA_NTm1xdxd, Xprime_NTm1xd,
                               tf.reshape(X[:,1:,:], [Nsamps*(NTbins-1), xDim])]


class NonLinear(NoisyEvolution):
    """
    """
    def __init__(self, X, params):
        """
        """
        NoisyEvolution.__init__(self, X, params)
        
        # Define the evolution for *this* instance
        self.nextX_NxTxd, self.nextXwinflow_NxTxd = self._define_evolution_network()
    
    def _define_evolution_network(self, Input=None):
        """
        """
        xDim = self.xDim
        if Input is None:
            Input = self.X
            Nsamps = self.Nsamps
            NTbins = self.NTbins
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
        
        alpha = self.alpha
        rangeB = self.params.initrange_B
        evnodes = 200
        Input = tf.reshape(Input, [Nsamps*NTbins, xDim])
        fully_connected_layer = FullLayer(collections=['EVOLUTION_PARS'])
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input, evnodes, 'sigmoid', 'full1',
                                          initializer=tf.random_uniform_initializer(-4.0, 4.0))
            full2 = fully_connected_layer(full1, evnodes//2, 'relu', 'full2',
                                          initializer=tf.orthogonal_initializer())
            output = fully_connected_layer(full2, xDim, nl='linear', scope='output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        nextX_NTxd = tf.add(Input, alpha*output, name='nextX')
        
        X_norms = tf.norm(Input, axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        nextXwinflow_NTxd = tf.transpose(fl_mod*tf.transpose(nextX_NTxd, [1, 0]) + 
                                         0.95*(1.0-fl_mod)*tf.transpose(Input, [1, 0]), [1, 0])
        nextX_NxTxd = tf.reshape(nextX_NTxd, [Nsamps, NTbins, xDim])
        nextXwinflow_NxTxd = tf.reshape(nextXwinflow_NTxd, [Nsamps, NTbins, xDim])
        return nextX_NxTxd, nextXwinflow_NxTxd

    def get_f_grads(self, xin_1x1xd=None):
        """
        """
        if xin_1x1xd is None: xin_1x1xd = self.x
        
        # f = f_i with i = 1,..,xDim. The gradients are along the -1 dim.
        singlef_d = tf.reshape(self._define_evolution_network(xin_1x1xd)[0], [self.xDim])
        grad_list_dxd = tf.squeeze(tf.stack([tf.gradients(fi, xin_1x1xd) 
                                             for fi in tf.unstack(singlef_d)]))
        jacobian_dxdxd = []
        for grad in tf.unstack(grad_list_dxd):
            jacobian_dxdxd.append(tf.squeeze(tf.stack([tf.gradients(g, xin_1x1xd) 
                                             for g in tf.unstack(grad)])))
        jacobian_dxdxd = tf.stack(jacobian_dxdxd)

        return grad_list_dxd, jacobian_dxdxd 


class NonLinearEvolution(NonLinear):
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
        """
        NonLinear.__init__(self, X, params)
                        
        self.logdensity_Xterms = self.compute_LogDensity_Xterms()
        
    def compute_LogDensity_Xterms(self, Input=None, with_inflow=False):
        """
        Computes the symbolic log p(X, Y).         
        Inputs:
        """
        xDim = self.xDim
        if Input is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            X = self.X
            nextX_NxTxd = self.nextX_NxTxd if not with_inflow else self.nextXwinflow_NxTxd 
        else:
            Nsamps = tf.shape(Input)[0]
            NTbins = tf.shape(Input)[1]
            nextXs = self._define_evolution_network(Input)
            nextX_NxTxd = nextXs[0] if not with_inflow else nextXs[1]
            X = Input

        nextX_NTm1xd = tf.reshape(nextX_NxTxd[:,:-1,:], [Nsamps, NTbins-1, xDim])
        resX_NTm1xd = tf.reshape(X[:,1:,:] - nextX_NTm1xd , [Nsamps*(NTbins-1), xDim])
        resX0_Nxd = X[:,0,:] - self.x0
        
        # L = -0.5*(resX_0^T·Q0^{-1}·resX_0) - 0.5*Tr[resX^T·Q^{-1}·resX] + 0.5*N*log(Det[Q0^{-1}])
        #     + 0.5*N*T*log(Det[Q^{-1}]) - 0.5*N*T*d_X*log(2*Pi)
        L0 = -0.5*tf.reduce_sum(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd), name='L0') 
        L1 = -0.5*tf.reduce_sum(resX_NTm1xd*tf.matmul(resX_NTm1xd, self.QInv_dxd), name='L1' )
        L2 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))*tf.cast(Nsamps, DTYPE)
        L3 = ( 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*
               tf.cast((NTbins-1)*Nsamps, DTYPE) )
        L4 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*xDim, DTYPE)
        
        LatentDensity = L0 + L1 + L2 + L3 + L4
        
        self.LX_summ = tf.summary.scalar('LogDensity_Xterms', LatentDensity)
        self.LX1_summ = tf.summary.scalar('LX1', L1)
                
        return LatentDensity, [L0, L1, L2, L3, L4]
        
    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, Xvar_name, Nsamps=100, NTbins=30, X0data=None, with_inflow=False,
                 path_mse_threshold=0.5, draw_plots=False):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        print('Sampling from latent dynamics...')
        xDim = self.xDim  
        Q0Chol = sess.run(self.Q0Chol_dxd)
        QChol = sess.run(self.QChol_dxd)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, xDim])
        x0scale = 15.0
        
        nextX_NxTxd = self.nextXwinflow_NxTxd if with_inflow else self.nextX_NxTxd
#         nextX_NTxd = tf.reshape(nextX_NxTxd, [Nsamps*NTbins, xDim])
        for samp in range(Nsamps):
            # needed to avoid paths that start too close to an attractor
            samp_norm = 0.0
            
            # lower path_mse_threshold to keep paths closer to trivial
            # trajectories, x = const.
            while samp_norm < path_mse_threshold:
                X_single_samp_Txd = np.zeros([NTbins, xDim])
                x0 = ( x0scale*np.dot(np.random.randn(xDim), Q0Chol) if 
                       X0data is None else X0data[samp] )
                X_single_samp_Txd[0] = x0
                
                noise_samps = np.random.randn(NTbins, self.xDim)
                for curr_tbin in range(NTbins-1):
                    curr_X_1x1xd = np.reshape(X_single_samp_Txd[curr_tbin:curr_tbin+1, :], [1, 1, xDim])
                    nextXval_1x1xd = sess.run(nextX_NxTxd, feed_dict={Xvar_name : curr_X_1x1xd})
                    X_single_samp_Txd[curr_tbin+1,:] = ( nextXval_1x1xd[0,0] +
                                                         np.dot(noise_samps[curr_tbin+1], QChol) )
            
                # Compute MSE and discard path is MSE < path_mse_threshold
                # (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(X_single_samp_Txd[tbin+1] -
                                                    X_single_samp_Txd[tbin]) for tbin in 
                                                    range(NTbins-1)])
                samp_norm = Xsamp_mse
        
            Xdata_NxTxd[samp,:,:] = X_single_samp_Txd
        
        if draw_plots:
            self.plot_2Dquiver_paths(sess, Xdata_NxTxd, Xvar_name, with_inflow=with_inflow,
                                     draw=True, pause=True)

        return Xdata_NxTxd        
    
    def eval_nextX(self, sess, Xdata, Xvar_name, with_inflow=False):
        """
        Given a symbolic array of points in latent space Xdata = [X0, X1,...,XT], \
        gives the prediction for the next time point
         
        This is required for the 2D quiver plots.
        
        Args:
            Xdata: Points in latent space at which the dynamics A(X) shall be
            determined.
            with_inflow: Should an inward flow from infinity be superimposed to A(X)?
        """
        nextX_NxTxd = self.nextXwinflow_NxTxd if with_inflow else self.nextX_NxTxd
        return sess.run(nextX_NxTxd, feed_dict={Xvar_name : Xdata})


if __name__ == '__main__':
    pass
        