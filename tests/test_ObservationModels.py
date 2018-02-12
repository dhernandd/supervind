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

import numpy as np

import tensorflow as tf

from LatEvModels import LocallyLinearEvolution
from ObservationModels import PoissonObs

class PoissonObsTest(tf.test.TestCase):
    """
    """
    rndints = np.random.randint(1000, size=100).tolist()

    yDim = 10
    xDim = 2
    
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        Y = tf.placeholder(tf.float64, [None, None, yDim], 'Y')
        
        lm = LocallyLinearEvolution(xDim, X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sampleX = lm.sample_X(sess, with_inflow=True, init_variables=True)
        
        mgen = PoissonObs(yDim, xDim, Y, X, lm)
        
    def test_simple(self):
        print('Test 1:')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Nsamps = sess.run(self.mgen.Nsamps, feed_dict={'X:0' : self.sampleX})
            print('Nsamps:', Nsamps)
            
    def test_eval_rate(self):
        rndint = self.rndints.pop()
        with tf.Session(graph=self.graph) as sess:
            np.random.seed(rndint)
            rate_NTxD = self.mgen.rate_NTxD
            sess.run(tf.global_variables_initializer())
            sample_rate = sess.run(rate_NTxD, feed_dict={'X:0' : self.sampleX})
#             print('Sample Y:', sample_rate)
            
    def test_eval_rate_winput(self):
        rndint = self.rndints.pop()
        with self.graph.as_default():
            Xinput = tf.placeholder(dtype=tf.float64, shape=[None, None, self.xDim],
                                    name='Xinput')
            rate_NTxD = self.mgen._define_rate(Xinput) 
        with tf.Session(graph=self.graph) as sess:
            np.random.seed(rndint)
            np.random.seed(rndint)
            sess.run(tf.global_variables_initializer())
            sample_rate = sess.run(rate_NTxD, feed_dict={'Xinput:0' : self.sampleX})
#             print('Sample Y:', sample_rate)
         
    def test_sample(self):
        rndint = self.rndints.pop()
        with tf.Session(graph=self.graph) as sess:
            np.random.seed(rndint)
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
                                               with_inflow=True)
#             print('Xdata:', Xdata)
#             print('Ydata:', Ydata)

    def test_compute_LogDensity_Yterms(self):

        LD = self.mgen.LogDensity
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
                                               with_inflow=True)
            LD = sess.run(LD, feed_dict={'X:0' : Xdata, 'Y:0' : Ydata})
            print('LD:', LD)

    
    def test_compute_LogDensity_Yterms_winput(self):
        """
        Test that we can add a new LogDensity to the graph with different inputs
        """
        with self.graph.as_default():
            XInput = tf.placeholder(dtype=tf.float64, shape=[None, None, self.xDim], 
                                    name='Xinput_1')
            LD_winput, _, _ = self.mgen.compute_LogDensity(XInput) 
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
                                               with_inflow=True)
            LD_winput = sess.run(LD_winput, feed_dict={'Xinput_1:0' : Xdata,
                                                'Y:0' : Ydata})
            print('LD:', LD_winput)



if __name__ == '__main__':
    tf.test.main()

