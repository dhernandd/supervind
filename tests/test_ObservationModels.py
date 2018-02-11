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

import numpy as np

import tensorflow as tf

from LatEvModels_new import LocallyLinearEvolution
from ObservationModels_new import PoissonObs
<<<<<<< HEAD

=======
>>>>>>> 4c5c36502666a34bd96835f1269d133585771520

class PoissonObsTest(tf.test.TestCase):
    """
    """
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.3, -1.4], [6.7, 8.9]]])
    xdata1 = np.array([[1.0, 2.0], [1.5, 1.5], [20.0, 25.0]])

    yDim = 10
    xDim = 2
    n_tsteps = len(Xdata1[0][0])
    
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        Y = tf.placeholder(tf.float64, [None, None, yDim], 'Y')
        
        lm = LocallyLinearEvolution(xDim, X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sampleX = lm.sample_X(sess, with_inflow=True)
        
        mgen = PoissonObs(yDim, xDim, Y, X, lm)
        sample_rate = mgen._define_rate(X)
        
    def test_simple(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Nsamps = sess.run(self.mgen.Nsamps, feed_dict={'X:0' : self.sampleX})
#             print('Nsamps:', Nsamps)
             
    def test_eval_rate(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sampleY = sess.run(self.sample_rate, feed_dict={'X:0' : self.sampleX})
#             print('Sample Y:', sampleY)
         
    def test_sample(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
                                               with_inflow=True)
#             print('Xdata:', Xdata)
#             print('Ydata:', Ydata)
         
    def test_compute_LogDensity_Yterms(self):
        with tf.Session(graph=self.graph) as sess:
            LD, LX, LY = self.mgen.compute_LogDensity(self.X, with_inflow=True)
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
                                               with_inflow=True, Nsamps=2, NTbins=30)
            rate_data = sess.run(self.sample_rate, feed_dict={'X:0' : Xdata})
            rate_data = np.reshape(rate_data, [2, 30, 10])
#             print(Ydata)
#             print(rate_data)
#             print(np.sum((Ydata - rate_data)**2))
            Ltot = sess.run([LD, LX, LY], feed_dict={'X:0' : Xdata, 
                                        'Y:0' : Ydata})
#             print('Ltot:', Ltot)
            

if __name__ == '__main__':
    tf.test.main()

