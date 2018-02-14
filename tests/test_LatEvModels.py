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

import sys
sys.path.append('../code/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


from LatEvModels import LocallyLinearEvolution

class LocallyLinearEvolutionTest(tf.test.TestCase):
    """
    Over and over, the algorithm will require to compute tensors A following a rule:
    
    A = f(X)

    where X can be different tensors. The main purpose of these tests is to
    check that the implementation of this aspect is working as desired
    """
    xDim = 2
    Xdata1 = np.random.randn(2, 10, 2)
    
    # For some reason unbeknownst to man, tensorflow has decided to set in stone
    # the numpy RandomState in its test classes. At least that is what I gather
    # from many tests after getting the same result over and over in experiments
    # that should have been in principle independent. Obviously, this is
    # undocumented, as it is undocumented how to change the seed to random. I
    # mean, why would anyone feel the need to document that?! Breathe. Hack
    # follows.
    rndints = np.random.randint(1000, size=100).tolist()

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        lm = LocallyLinearEvolution(xDim, X)
        
        # Let's sample from X for later use.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sampleX = lm.sample_X(sess, with_inflow=True, Nsamps=2, 
                                  draw_plots=False, init_variables=False)

    def test_simple(self):
        """This test just checks that tensorflow variables are properly defined"""
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
  
            Nsamps = sess.run(self.lm.Nsamps, feed_dict={'X:0' : self.Xdata1})
            QInvChol = sess.run(self.lm.QInvChol_dxd, feed_dict={'X:0' : self.Xdata1})
            QChol = sess.run(self.lm.QChol_dxd, feed_dict={'X:0' : self.Xdata1})
            QInv = sess.run(self.lm.QInv_dxd, feed_dict={'X:0' : self.Xdata1})
            Q0Inv = sess.run(self.lm.Q0Inv_dxd, feed_dict={'X:0' : self.Xdata1})
#             print('Nsamps:', Nsamps)
#             print('QInvChol:', QInvChol)
#             print('QChol:', QChol)
#             print('QInv:', QInv)
#             print('Q0Inv:', Q0Inv)
#             print('\n\n')
              
    def test_simple2(self):
        """This test just checks that things are properly defined"""
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Alinear = sess.run(self.lm.Alinear_dxd)
            alpha = sess.run(self.lm.alpha)
#             print('Alinear:', Alinear)
#             print('alpha:', alpha)
#             print('\n\n')

    def test_evalA(self):
        """Is A evaluating correctly?"""
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            A = sess.run(self.lm.A_NxTxdxd, feed_dict={'X:0' : self.Xdata1})
#             print('A:', A)
            print('A.shape:', A.shape)

    def test_evalsymbA(self):
        """
        Tests that a different A can later on be added to the graph by passing
        an argument to the method _define_evolution_network.
        """
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            newX = tf.placeholder(dtype=tf.float64, shape=[None, 5, 2], name='newX')

            # add a new A to the graph
            symbA, symbAwinflow = self.lm._define_evolution_network(newX)
            newXdata = np.random.randn(3,5,2)
            A = sess.run(symbA, feed_dict={'newX:0' : newXdata})
            Awinflow = sess.run(symbAwinflow, feed_dict={'newX:0' : newXdata})
#             print('A:', A)
#             print('A:', Awinflow)
            print('Shapes:', A.shape, Awinflow.shape)

    def test_sampleX(self):
        """Tests the sampling method"""
        print('\nTest')
        sess = tf.Session(graph=self.graph)
        with sess:
            Xdata = self.lm.sample_X(sess, with_inflow=True,
                                   init_variables=True)
            print('Xdata.shape:', Xdata.shape)
            print(Xdata[0])
 
    def test_compute_LD(self):
        """
        Evaluates the cost from the latent evolution instance. 
        """
        print('\nTest')
        rndint = self.rndints.pop()
        print('seed:',rndint)
        sess = tf.Session(graph=self.graph)
        with sess:
            np.random.seed(rndint)
            sess.run(tf.global_variables_initializer())
            Xdata = self.lm.sample_X(sess, with_inflow=True, draw_plots=False,
                                     init_variables=False)
            LD1, arrays = self.lm.logdensity_Xterms
            LD_val = sess.run(LD1, feed_dict={'X:0' : Xdata})
            A, M, B, C = sess.run([arrays[0], arrays[1],arrays[2], arrays[3]], feed_dict={'X:0' : Xdata})
#             print('Xdata:', Xdata[0][:5])
#             for i in range(len(A)):
#                 print('M:', M[i])
#                 print('A*M:',np.dot(A[i], M[i]))
#                 print('B:', B[i])
#                 print('C:', C[i])
#                 print('B-C:', B[i]-C[i])
            print('LD:', LD_val)
 
    def test_compute_LD_winput(self):
        """
        Tests that a different LogDensity node, with a different input, can be
        added to the graph later on demand.
         
        Also, evaluates the LogDensity both on data generated through this
        network and generated via a different network. The component L2 of the
        LogDensity should be much smaller in the first case, since in that case,
        the same A that was used for generation is being used to compute the LD.
        """
        print('\nTest')
        rndint = self.rndints.pop()
        print('seed', rndint)
        with self.graph.as_default():
            Xinput = tf.placeholder(dtype=tf.float64, shape=[None, None, self.xDim],
                                    name='Xinput')
            LD2, arrays = self.lm.compute_LogDensity_Xterms(Xinput, with_inflow=True)
            
        sess = tf.Session(graph=self.graph)
        with sess:
            np.random.seed(rndint)
            sess.run(tf.global_variables_initializer())
            Xdata = self.lm.sample_X(sess, with_inflow=True, draw_plots=False,
                                     init_variables=False, Nsamps=2)
            LD_val = sess.run(LD2, feed_dict={'Xinput:0' : Xdata})
            L1, L2, res =  sess.run([arrays[0], arrays[1], arrays[2]], 
                                  feed_dict={'Xinput:0' : Xdata})
            LD2_val = sess.run(LD2, feed_dict={'Xinput:0' : self.sampleX})
            L1a, L2a, resa =  sess.run([arrays[0], arrays[1], arrays[2]], 
                                  feed_dict={'Xinput:0' : self.sampleX})
#             print('Xdata:', Xdata[0][:5])
#             for i in range(len(A)):
#                 print('M:', M[i])
#                 print('A*M:',np.dot(A[i], M[i]))
#                 print('B:', B[i])
#                 print('C:', C[i])
#                 print('B-C:', B[i]-C[i])
            print('LD:', LD_val)
            print('LDa:', LD2_val)
            print(L2, '<<', L1)
# 
#     def test_compute_grads(self):
#         print(self.lm.Agrads_d2xd)


if __name__ == '__main__':
    tf.test.main()



