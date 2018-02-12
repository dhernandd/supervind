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
from ObservationModels import PoissonObs
from RecognitionModels_new import SmoothingNLDSTimeSeries


class SmoothingNLDSTimeSeries(tf.test.TestCase):
    """
    """
    yDim = 10
    xDim = 2
        
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float64, [None, None, xDim], 'X')
        Y = tf.placeholder(tf.float64, [None, None, yDim], 'Y')
        
        lm = LocallyLinearEvolution(xDim, X)        
        mgen = PoissonObs(yDim, xDim, Y, X, lm)
        mrec = SmoothingNLDSTimeSeries(yDim, xDim, Y, X, lm)
        
        # Let us first generate some data that we may use
        with tf.Session() as sess:
            Ydata, Xdata = mgen.sample_XY(sess, with_inflow=True, Nsamps=3, NTbins=4)
    
    
    def test_simple(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Mu = sess.run(self.mrec.Mu_NTxd, feed_dict={'X:0' : self.Xdata,
                                                        'Y:0' : self.Ydata})            
            LambdaMu = sess.run(self.mrec.LambdaMu_NxTxd, feed_dict={'X:0' : self.Xdata,
                                                                     'Y:0' : self.Ydata})
            print('Mu:', Mu)
            print('LambdaMu:', LambdaMu)
            
    def test_postX(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            postX = sess.run(self.mrec.postX, feed_dict={'Y:0' : self.Ydata,
                                                         'X:0' : self.Xdata})
            print(postX, postX.shape)
            TheChol = sess.run(self.mrec.TheChol_2xNxTxdxd, feed_dict={'Y:0' : self.Ydata,
                                                           'X:0' : self.Xdata})
#             AA = sess.run(self.mrec.AA_NxTxdxd, feed_dict={'Y:0' : self.Ydata,
#                                                            'X:0' : self.Xdata})
            print(TheChol[0].shape)
            print(TheChol[0])
            LogDet = sess.run(self.mrec.LogDet, feed_dict={'Y:0' : self.Ydata,
                                                           'X:0' : self.Xdata})
            print('LogDet', LogDet)

    def test_sample_postX(self):
        with self.graph.as_default():
            samps = self.mrec.sample_postX()
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            samps_nmric = sess.run(samps, feed_dict={'Y:0' : self.Ydata,
                                                     'X:0' : self.Xdata})
            print('Samps numeric', samps_nmric)
        
    
    
    
if __name__ == '__main__':
    tf.test.main()
