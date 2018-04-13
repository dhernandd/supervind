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
# Jupyter notebook. A fairy dies in Neverland every time you run this.
if __name__ == 'LatEvModels':
    from datetools import addDateTime #@UnresolvedImport #@UnusedImport
    from layers import FullLayer #@UnresolvedImport #@UnusedImport
else:
    from .datetools import addDateTime # @UnresolvedImport @Reimport
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
    def __init__(self, X, params, Ids):
        """
        Args:
            xDim:
            X:
            nnname:
        """        
        self.X = X
        self.params = params
        self.Ids = Ids = tf.placeholder(dtype=tf.int32, shape=[None], name='Ids') if Ids is None else Ids

        self.xDim = xDim = params.xDim
        self.pDim = pDim = params.pDim
        self.rDim = xDim + pDim
        
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]

        if hasattr(params, 'num_diff_entities'): self.num_diff_entities = params.num_diff_entities
        else: self.num_diff_entities = 1
        self.ev_params_Pxp = tf.get_variable('ev_params', shape=[self.num_diff_entities, pDim])

        # Variance (Q) of the state-space evolution. 
        init_Q = params.init_Q
        self.QInvChol_dxd = tf.get_variable('QInvChol', 
                                            initializer=tf.cast(init_Q*tf.eye(xDim), DTYPE),
                                            trainable=params.is_Q_trainable)
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                              name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True,
                           name='Q')
        
        # Variance of the initial points
        init_Q0 = params.init_Q0
        self.Q0InvChol_dxd = tf.get_variable('Q0InvChol',
                                             initializer=tf.cast(init_Q0*tf.eye(xDim), DTYPE))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd,
                                   transpose_b=True, name='Q0Inv')
        
        # The mean starting coordinates in state-space
        self.x0 = tf.get_variable('x0', initializer=tf.cast(tf.zeros(self.xDim), DTYPE))
         
        self.alpha = params.alpha
#         init_alpha = tf.constant_initializer(params.alpha)
#         self.alpha = tf.get_variable('alpha', shape=[], initializer=init_alpha,
#                                          dtype=DTYPE, trainable=False)

#       Define base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear_dxd = tf.get_variable('Alinear', initializer=tf.eye(xDim),
                                               dtype=DTYPE)
        
        # Define the evolution for *this* instance
        self.A_NxTxdxd, self.Awinflow_NxTxdxd, self.B_NxTxdxd = self._define_evolution_network()
    
    def _define_evolution_network(self, Input=None, Ids=None):
        """
        """
        xDim = self.xDim
        pDim = self.pDim
        rDim = xDim + pDim
        
        if Input is None and Ids is not None: raise ValueError("Must provide an Input for these Ids")
        X_NxTxd = self.X if Input is None else Input
        if Ids is None: Ids = self.Ids            

        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        
        ev_params_Nxp = tf.gather(self.ev_params_Pxp, indices=Ids)
        ev_params_NxTxp = tf.tile(tf.expand_dims(ev_params_Nxp, axis=1), [1, NTbins, 1])

        rangeB = self.params.initrange_B
        evnodes = 200
        Input_NxTxr = tf.concat([X_NxTxd, ev_params_NxTxp], axis=2)
        Input_NTxr = tf.reshape(Input_NxTxr, [Nsamps*NTbins, rDim])
        fully_connected_layer = FullLayer(collections=['EVOLUTION_PARS'])
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(Input_NTxr, evnodes, 'softmax', 'full1')
            full2 = fully_connected_layer(full1, evnodes//2, 'softplus', 'full2',
                                          initializer=tf.orthogonal_initializer())
            output = fully_connected_layer(full2, xDim**2, nl='linear', scope='output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim])
        
        A_NTxdxd = self.alpha*B_NTxdxd + self.Alinear_dxd # Broadcast
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

    def get_A_grads(self, xin=None, idin=None):
        xDim = self.xDim
        if xin is None: xin = self.x

        singleA_1x1xdxd = self._define_evolution_network(xin, idin)[0]
        singleA_d2 = tf.reshape(singleA_1x1xdxd, [xDim**2])
        grad_list_d2xd = tf.squeeze(tf.stack([tf.gradients(Ai, xin) for Ai
                                              in tf.unstack(singleA_d2)]))

        return grad_list_d2xd 

    def eval_nextX(self, session, Xdata, Xvar_name, with_inflow=False, Id=0):
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
        
        Iddata = np.full(len(Xdata), Id, dtype=np.int32)
        totalA = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        A = session.run(totalA, feed_dict={Xvar_name : Xdata,
                                           'VAEC/Ids:0' : Iddata})
        A = A[:,:-1,:,:].reshape(Nsamps*(Tbins-1), self.xDim, self.xDim)
        Xdata = Xdata[:,:-1,:].reshape(Nsamps*(Tbins-1), self.xDim)
                
        return np.einsum('ij,ijk->ik', Xdata, A).reshape(Nsamps, Tbins-1, self.xDim)
    
    @staticmethod
    def define2DLattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(2, -1).T

    def quiver2D_flow(self, session, Xvar_name, clr='black', scale=25,
                      x1range=(-35.0, 35.0), x2range=(-35.0, 35.0), figsize=(13,13), 
                      pause=False, draw=False, with_inflow=False, newfig=True, savefile=None,
                      Id=0):
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
        
        nextX = self.eval_nextX(session, lattice, Xvar_name, with_inflow=with_inflow, Id=Id)
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
                            pause=False, skipped=1, feed_range=True, range_xs=20.0, Id=0):
        """
        """
        if savefig:
            if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
            rslt_file = rlt_dir + rslt_file
        
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata, pause=pause, draw=draw, newfig=True, skipped=skipped)
        if feed_range:
            x1range, x2range = axes.get_xlim(), axes.get_ylim()
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        else:
            x1range = x2range = (-range_xs, range_xs)
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        
        self.quiver2D_flow(session, Xvar_name, pause=pause, x1range=x1range, 
                           x2range=x2range, scale=s, newfig=False, 
                           with_inflow=with_inflow, draw=draw, Id=Id)
        if savefig:
            plt.savefig(rslt_file)
        else:
            pass
        plt.close()


class LocallyLinearEvolution(NoisyEvolution):
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, X, params, Ids=None):
        """
        """
        NoisyEvolution.__init__(self, X, params, Ids=Ids)
                        
        self.logdensity_Xterms, self.checks_LX = self.compute_LogDensity_Xterms()
        
        
    def compute_LogDensity_Xterms(self, Input=None, with_inflow=False):
        """
        Computes the symbolic log p(X, Y).         
        Inputs:
        """
        xDim = self.xDim
        X_NxTxd = self.X if Input is None else Input
        if Input is None:
            A_NxTxdxd = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        else:
            A_NxTxdxd, Awinflow_NxTxdxd, _ = self._define_evolution_network(X_NxTxd)
            A_NxTxdxd = A_NxTxdxd if not with_inflow else Awinflow_NxTxdxd
        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        
        Xprime_NxTm1xd = tf.squeeze(tf.matmul(tf.expand_dims(X_NxTxd[:,:-1], axis=2),
                                              A_NxTxdxd[:,:-1]))
        resX_NxTm1x1xd = tf.expand_dims(X_NxTxd[:,1:] - Xprime_NxTm1xd, axis=2)
        QInv_NxTm1xdxd = tf.tile(tf.reshape(self.QInv_dxd, [1, 1, xDim, xDim]),
                                 [Nsamps, NTbins-1, 1, 1])
        resX0_Nxd = X_NxTxd[:,0] - self.x0

        LX1 = -0.5*tf.reduce_sum(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd), name='LX0')
        LX2 = -0.5*tf.reduce_sum(resX_NxTm1x1xd*tf.matmul(resX_NxTm1x1xd, QInv_NxTm1xdxd), name='L2')
        LX3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))
        LX4 = 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*tf.cast((NTbins-1), DTYPE)
        LX5 = -0.5*np.log(2*np.pi)*tf.cast(NTbins*xDim, DTYPE)
        
        LatentDensity = LX1 + LX2 + LX3 + LX4 + LX5
        
        return LatentDensity, [LX1, LX2, LX3, LX4, LX5]
    

    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, Xvar_name, Nsamps=2, NTbins=3, X0data=None, with_inflow=False,
                 path_mse_threshold=0.1, draw_plots=False, init_variables=True):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        print('Sampling from latent dynamics...')
        if init_variables: 
            sess.run(tf.global_variables_initializer())
        
        xDim = self.xDim
        Q0Chol = sess.run(self.Q0Chol_dxd)
        QChol = sess.run(self.QChol_dxd)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, self.xDim])
        x0scale = 15.0
        
        A_NxTxdxd = self.Awinflow_NxTxdxd if with_inflow else self.A_NxTxdxd
        A_NTxdxd = tf.reshape(A_NxTxdxd, shape=[-1, xDim, xDim])
        for samp in range(Nsamps):
            samp_norm = 0.0 # needed to avoid paths that start too close to an attractor
            
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


        