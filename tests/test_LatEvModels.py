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

import sys
sys.path.append('../code/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


from LatEvModels import LocallyLinearEvolution



class LocallyLinearEvolutionTest(tf.test.TestCase):
    """
    """
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.3, -1.4], [6.7, 8.9]]])
    xdata1 = np.array([[1.0, 2.0], [1.5, 1.5], [20.0, 25.0]])

    xDim = 2
    n_tsteps = len(Xdata1[0][0])
    
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        lm = LocallyLinearEvolution(xDim, X)
        sampleX = lm.sample_X(with_inflow=True)

    def test_simple(self):
        print('Test 1:')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            Nsamps = sess.run(self.lm.Nsamps, feed_dict={'X:0' : self.Xdata1})
            QInvChol = sess.run(self.lm.QInvChol_dxd, feed_dict={'X:0' : self.Xdata1})
            QChol = sess.run(self.lm.QChol_dxd, feed_dict={'X:0' : self.Xdata1})
            QInv = sess.run(self.lm.QInv_dxd, feed_dict={'X:0' : self.Xdata1})
            Q0Inv = sess.run(self.lm.Q0Inv_dxd, feed_dict={'X:0' : self.Xdata1})
            print('Nsamps:', Nsamps)
            print('QInvChol:', QInvChol)
            print('QChol:', QChol)
            print('QInv:', QInv)
            print('Q0Inv:', Q0Inv)
            print('\n\n')
            
    def test_simple2(self):
        print('Test 2:')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Alinear = sess.run(self.lm.Alinear_dxd)
            alpha = sess.run(self.lm.alpha)
            print('Alinear:', Alinear)
            print('alpha:', alpha)
            print('\n\n')
             
    def test_evalB(self):
        print('Test 3:')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            B = sess.run(self.lm.B_NTxdxd, feed_dict={'X:0' : self.Xdata1})
            print('B:', B)
             
    def test_evalA(self):
        print('Test 4:')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            A = sess.run(self.lm.A_NTxdxd, feed_dict={'X:0' : self.Xdata1})
            print('A:', A)
             
    def test_sampleX(self):
        print('Test 5:')
        with tf.Session(graph=self.graph) as sess:
            print(self.lm.sample_X(with_inflow=True))
         
    def test_sampleX2(self):
        print('Test 6:')
        with tf.Session(graph=self.graph) as sess:
            print(self.lm.sample_X(with_inflow=True, draw_plots=True))
        
    def test_computeLogDensity(self):
        print('Test 7:')
        with tf.Session(graph=self.graph) as sess:
            logdensity, _ = self.lm.compute_LogDensity_Xterms()
            sess.run(tf.global_variables_initializer())
            print('LogDensity Xterms:', sess.run(logdensity, feed_dict={'X:0' : self.Xdata1}))

#     def test_evalnextX(self):
#         print('Test 6:')
#         print(self.lm.eval_nextX(self.sampleX, withInflow=True))
        
#     def test_quiver2D_flow(self):
#         self.lm.quiver2D_flow(withInflow=True)
#         self.lm.quiver2D_flow(withInflow=True)
        
#     def test_plot2D(self):
#         self.lm.plot_2Dquiver_paths(self.sampleX, withInflow=False)
        

if __name__ == '__main__':
    tf.test.main()



