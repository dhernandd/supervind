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
import sys
sys.path.append('../code/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from code.LLinearEv_wParams import LocallyLinearEvolution_wParams

DTYPE = tf.float32

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_integer('yDim', 50, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_integer('pDim', 1, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 1.0, "")
flags.DEFINE_float('init_Q', 2.0, "")
flags.DEFINE_float('alpha', 0.4, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
flags.DEFINE_integer('num_diff_entities', 2, "")
flags.DEFINE_integer('batch_size', 5, "")
params = tf.flags.FLAGS

class LocallyLinearEv_wParamsTest(tf.test.TestCase):
    """
    Over and over, the algorithm will require to compute tensors A following a rule:
    
    A = f(X)

    where X can be different tensors. The main purpose of these tests is to
    check that the implementation of this aspect is working as desired
    """
    seed_list = np.random.randint(1000, size=100).tolist()
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('LM1'):
                X1 = tf.placeholder(DTYPE, [1, None, params.xDim], 'X1')
                lm1 = LocallyLinearEvolution_wParams(X1, params)
                
                # Let's sample from X for later use.
                sess.run(tf.global_variables_initializer())
                sampleX1, IdX1 = lm1.sample_X(sess, with_inflow=True)
                
#     def test_evalA(self):
#         """Is A evaluating correctly?"""
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             A = sess.run(self.lm.A_NxTxrxr, feed_dict={'X:0' : self.Xdata1})
# #             print('A:', A)
#             print('A.shape:', A.shape)
#  
#     def test_evalsymbA(self):
#         """
#         Tests that a different A can later on be added to the graph by passing
#         an argument to the method _define_evolution_network.
#         """
#         xDim = self.xDim
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             newX = tf.placeholder(dtype=tf.float64, shape=[None, 5, xDim], name='newX')
#  
#             # add a new A to the graph
#             symbA, symbAwinflow = self.lm._define_evolution_network(newX)
#             newXdata = np.random.randn(3, 5, xDim)
#             A = sess.run(symbA, feed_dict={'newX:0' : newXdata})
#             Awinflow = sess.run(symbAwinflow, feed_dict={'newX:0' : newXdata})
# #             print('A:', A)
# #             print('A:', Awinflow)
#             print('Shapes:', A.shape, Awinflow.shape)
 
#     def test_sampleX(self):
#         """Tests the sampling method"""
#         print('\nTest')
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             Xdata = self.lm.sample_X(sess, with_inflow=True, 
#                                      init_variables=True)
#             print('Xdata.shape:', Xdata.shape)
#             print(Xdata[0])
  
#     def test_compute_LD(self):
#         """
#         Evaluates the cost from the latent evolution instance. 
#         """
#         print('\nTest')
#         rndint = self.rndints.pop()
#         print('seed:',rndint)
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             np.random.seed(rndint)
#             sess.run(tf.global_variables_initializer())
#             Xdata = self.lm.sample_X(sess, with_inflow=True, draw_plots=False,
#                                      init_variables=False)
#             LD1, arrays = self.lm.logdensity_Xterms
#             LD_val = sess.run(LD1, feed_dict={'X:0' : Xdata})
#             L1, L2, L3, L4 = sess.run([arrays[0], arrays[1],arrays[2], arrays[3]], feed_dict={'X:0' : Xdata})
# 
#             print('LD:', LD_val)
 
#     def test_compute_LD_winput(self):
#         """
#         Tests that a different LogDensity node, with a different input, can be
#         added to the graph later on demand.
#           
#         Also, evaluates the LogDensity both on data generated through this
#         network and generated via a different network. The component L2 of the
#         LogDensity should be much smaller in the first case, since in that case,
#         the same A that was used for generation is being used to compute the LD.
#         """
#         print('\nTest')
#         rndint = self.rndints.pop()
#         print('seed', rndint)
#         with self.graph.as_default():
#             Xinput = tf.placeholder(dtype=tf.float64, shape=[None, None, self.xDim],
#                                     name='Xinput')
#             LD2, arrays = self.lm.compute_LogDensity_Xterms(Xinput, with_inflow=True)
#              
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             np.random.seed(rndint)
#             sess.run(tf.global_variables_initializer())
#             Xdata = self.lm.sample_X(sess, with_inflow=True, draw_plots=False,
#                                      init_variables=False, Nsamps=2)
#             LD_val = sess.run(LD2, feed_dict={'Xinput:0' : Xdata})
#             L1, L2, res =  sess.run([arrays[0], arrays[1], arrays[2]], 
#                                   feed_dict={'Xinput:0' : Xdata})
#             LD2_val = sess.run(LD2, feed_dict={'Xinput:0' : self.sampleX})
#             L1a, L2a, resa =  sess.run([arrays[0], arrays[1], arrays[2]], 
#                                   feed_dict={'Xinput:0' : self.sampleX})
# 
#             print('LD:', LD_val)
#             print('LDa:', LD2_val)
#             print(abs(L2), '<<', abs(L1))

 
#     def test_compute_grads(self):
#         sess = tf.Session(graph=self.graph)
#         with sess:
#             sess.run(tf.global_variables_initializer())
#             xdata = self.lm.sample_X(sess, with_inflow=True, draw_plots=False,
#                                      init_variables=False, Nsamps=1, NTbins=1)
#             print('\nA grads:', 
#                   sess.run(self.lm.Agrads_r2xd, feed_dict={'x:0' : xdata}) ) 

if __name__ == '__main__':
    tf.test.main()


