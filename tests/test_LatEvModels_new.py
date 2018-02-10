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


from LatEvModels_new import LocallyLinearEvolution


class LocallyLinearEvolutionTest(tf.test.TestCase):
    """
    """
    xDim = 2
    Xdata1 = np.random.randn(2, 10, 2)
        
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        lm = LocallyLinearEvolution(xDim, X)
#         sampleX = lm.sample_X(with_inflow=True)

    def test_simple(self):
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
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Alinear = sess.run(self.lm.Alinear_dxd)
            alpha = sess.run(self.lm.alpha)
#             print('Alinear:', Alinear)
#             print('alpha:', alpha)
#             print('\n\n')
#              
              
    def test_evalA(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            A = sess.run(self.lm.A_NTxdxd, feed_dict={'X:0' : self.Xdata1})
#             print('A:', A)
            

    def test_evalsymbA(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            newX = tf.placeholder(dtype=tf.float64, shape=[None, 5, 2], name='newX')
            symbA = self.lm._define_evolution_network(newX)
            newXdata = np.random.randn(3,5,2) 
            A = sess.run(symbA, feed_dict={'newX:0' : newXdata})
#             print(A.shape)
#             print('A:', A)
            
    def test_sampleX(self):
        print('Test 5:')
        sess = tf.Session(graph=self.graph)
        with sess:
            print(self.lm.sample_X(sess, with_inflow=True, 
                                   init_variables=False))
#           
#     def test_sampleX2(self):
#         print('Test 6:')
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             print(self.lm.sample_X(sess, with_inflow=True, draw_plots=True))
#          
#     def test_computeLogDensity(self):
#         print('Test 7:')
#         with tf.Session(graph=self.graph) as sess:
#             logdensity, _ = self.lm.compute_LogDensity_Xterms()
#             sess.run(tf.global_variables_initializer())
#             print('LogDensity Xterms:', sess.run(logdensity, 
#                                                  feed_dict={'X:0' : self.Xdata1}))
#  
#     def test_sampleX3(self):
#         print('Test 8')
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             print(self.lm.sample_X(sess, with_inflow=True, draw_plots=True))


        

if __name__ == '__main__':
    tf.test.main()



