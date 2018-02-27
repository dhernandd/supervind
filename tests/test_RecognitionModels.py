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

from code.LatEvModels import LocallyLinearEvolution
from code.ObservationModels import PoissonObs
from code.RecognitionModels import SmoothingNLDSTimeSeries

DTYPE = tf.float32

class SmoothingNLDSTimeSeriesTest(tf.test.TestCase):
    """
    """
    seed_list = np.random.randint(1000, size=100).tolist()

    yDim = 10
    xDim = 2
    Nsamps = 100
    NTbins = 50
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('M1'):
                X1 = tf.placeholder(DTYPE, [None, None, xDim], 'X1')
                Y1 = tf.placeholder(DTYPE, [None, None, yDim], 'Y1')
                mrec1 = SmoothingNLDSTimeSeries(yDim, xDim, Y1, X1) 
                lm1 = mrec1.get_lat_ev_model()
                LD1_winflow, _ = lm1.compute_LogDensity_Xterms(X1, with_inflow=True) 
                
                mgen1 = PoissonObs(yDim, xDim, Y1, X1, lm1, is_out_positive=True)

                ld1, checks1 = mgen1.compute_LogDensity(with_inflow=True)
            with tf.variable_scope('M2'):
                X2 = tf.placeholder(DTYPE, [None, None, xDim], 'X2')
                Y2 = tf.placeholder(DTYPE, [None, None, yDim], 'Y2')
                mrec2 = SmoothingNLDSTimeSeries(yDim, xDim, Y2, X2) 
                lm2 = mrec2.get_lat_ev_model()
                LD2_winflow, _ = lm2.compute_LogDensity_Xterms(X2, with_inflow=True,)
                
                mgen2 = PoissonObs(yDim, xDim, Y2, X2, lm2, is_out_positive=True)
            
            Xtest = tf.placeholder(DTYPE, [1, 3, xDim], 'Xtest')
            Ytest = tf.placeholder(DTYPE, [1, 3, yDim], 'Ytest')
            Mu_test, Lambda_test, LambdaMu_test = mrec1.get_Mu_Lambda(Ytest)
            TheChol_test, postX_test, checks = mrec1._compute_TheChol_postX(Xtest)
            
            # Let's sample from X, Y.
            sess.run(tf.global_variables_initializer())
            sampleY1, sampleX1 = mgen1.sample_XY(sess, Xvar_name='M1/X1:0', Nsamps=Nsamps,
                                                 NTbins=NTbins, with_inflow=True)
            sampleY2, sampleX2 = mgen2.sample_XY(sess, Xvar_name='M2/X2:0', Nsamps=Nsamps,
                                                 NTbins=NTbins, with_inflow=True)
    
    
    def test_TheChol(self):
        with self.sess.as_default():
            sampleY1, sampleX1 = self.mgen1.sample_XY(self.sess, Xvar_name='M1/X1:0', Nsamps=1,
                                                      NTbins=3, with_inflow=True)            
            Mu_test_val = self.sess.run(self.Mu_test, feed_dict={'Ytest:0' : sampleY1})
            Lambda_test_val = self.sess.run(self.Lambda_test, feed_dict={'Ytest:0' : sampleY1})
            LambdaMu_test_val = self.sess.run(self.LambdaMu_test, feed_dict={'Ytest:0' : sampleY1})
            print('Mu, Lambda, LambdaMu:', Mu_test_val, Lambda_test_val, LambdaMu_test_val)
            
            
    def test_MuLambda_statistics(self):
        with self.sess.as_default():
            Mu = self.sess.run(self.mrec1.Mu_NxTxd, feed_dict={'M1/Y1:0' : self.sampleY1})
            Lambda = self.sess.run(self.mrec1.Lambda_NxTxdxd, feed_dict={'M1/Y1:0' : self.sampleY1})
            LambdaMu = self.sess.run(self.mrec1.LambdaMu_NxTxd, feed_dict={'M1/Y1:0' : self.sampleY1})
            print('Mu (mean, rate, max):', np.mean(Mu), np.std(Mu), np.max(Mu))
            print('Lambda (mean, rate, max)', np.mean(Lambda), np.std(Lambda), np.max(Lambda))
            print('LambdaMu:', np.mean(LambdaMu), np.std(LambdaMu), np.max(LambdaMu))
            print('\n')
             
#     def test_postX(self):
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             postX = sess.run(self.mrec.postX, feed_dict={'Y:0' : self.Ydata,
#                                                          'X:0' : self.Xdata})
# #             print(postX, postX.shape)
#             TheChol = sess.run(self.mrec.TheChol_2xNxTxdxd, feed_dict={'Y:0' : self.Ydata,
#                                                            'X:0' : self.Xdata})
# #             AA = sess.run(self.mrec.AA_NxTxdxd, feed_dict={'Y:0' : self.Ydata,
# #                                                            'X:0' : self.Xdata})
# #             print(TheChol[0].shape)
# #             print(TheChol[0])

#     def test_Entropy(self):
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             Entropy = sess.run(self.mrec.Entropy, feed_dict={'Y:0' : self.Ydata,
#                                                              'X:0' : self.Xdata})
#             print('Entropy', Entropy)
 
#     def test_Entropy_winput(self):
#         rndint = self.rndints.pop()
#         with self.graph.as_default():
#             XInput = tf.placeholder(dtype=tf.float64, shape=[None, None, self.xDim], 
#                                     name='Xinput_1')
#             Ewinput = self.mrec.compute_Entropy(XInput) 
#         with tf.Session(graph=self.graph) as sess:
#             np.random.seed(rndint)
#             sess.run(tf.global_variables_initializer())
#             Ydata, Xdata = self.mgen.sample_XY(sess, init_variables=False,
#                                                with_inflow=True)
#             Ewinput = sess.run(Ewinput, feed_dict={'Xinput_1:0' : Xdata,
#                                                    'Y:0' : Ydata})
#             print('E:', Ewinput)

#     def test_grad(self):
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             gterm = sess.run(self.mrec.gradterm_postX_NxTxd, 
#                              feed_dict={'Y:0' : self.Ydata, 'X:0' : self.Xdata})
#             print('Grad term', gterm)
        
    
    
    
if __name__ == '__main__':
    tf.test.main()
