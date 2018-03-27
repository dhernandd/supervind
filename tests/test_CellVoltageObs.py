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
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from code.LLinearEv_wParams import LocallyLinearEvolution_wParams 
from code.ObservationModels import CellVoltageObs

DTYPE = tf.float32

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_integer('yDim', 1, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_integer('pDim', 1, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 1.0, "")
flags.DEFINE_float('init_Q', 2.0, "")
flags.DEFINE_float('alpha', 0.5, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
flags.DEFINE_float('initrange_Goutmean', 0.03,"")
flags.DEFINE_float('initrange_Goutvar', 1e-1,"")
flags.DEFINE_float('initbias_Goutmean', 1.0,"")
flags.DEFINE_integer('num_diff_entities', 2, "")
params = tf.flags.FLAGS


class CellVoltageObsTest(tf.test.TestCase):
    """
    """
    Nsamps = 100
    NTbins = 50
    xDim = params.xDim

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('M1'):
                X1 = tf.placeholder(DTYPE, [1, None, xDim], 'X1')
                Y1 = tf.placeholder(DTYPE, [None, None, 1], 'Y1')
                
                lm1 = LocallyLinearEvolution_wParams(X1, params)
                LD1_winflow, _ = lm1.compute_LogDensity_Xterms(X1, with_inflow=True) 
                
                mgen1 = CellVoltageObs(Y1, X1, params, lm1, is_out_positive=True)
                ld1, checks1 = mgen1.compute_LogDensity(with_inflow=True)
                sess.run(tf.global_variables_initializer())
                sampleY1, sampleX1, Ids = mgen1.sample_XY(sess, feed_key='M1/X1:0', Nsamps=Nsamps,
                                                          NTbins=50, with_inflow=True)
                
                plt.figure(figsize=(8, 8))
                plt.plot(sampleY1[0,:,0])
                plt.plot(sampleX1[0,:,0])
                plt.plot(sampleX1[0,:,1])
                plt.show()
#             with tf.variable_scope('M2'):
#                 X2 = tf.placeholder(DTYPE, [None, None, xDim], 'X2')
#                 Y2 = tf.placeholder(DTYPE, [None, None, yDim], 'Y2')
#                 
#                 lm2 = LocallyLinearEvolution(X2, params)
#                 LD2_winflow, _ = lm2.compute_LogDensity_Xterms(X2, with_inflow=True,)
#                  
#                 mgen2 = PoissonObs(Y2, X2, params, lm2, is_out_positive=True)
#             with tf.variable_scope('M3'):
#                 X3 = tf.placeholder(DTYPE, [None, None, xDim], 'X3')
#                 Y3 = tf.placeholder(DTYPE, [None, None, yDim], 'Y3')
#                 
#                 lm3 = LocallyLinearEvolution(X3, params)
#                 LD3_winflow, _ = lm3.compute_LogDensity_Xterms(X3, with_inflow=True,)
#                  
#                 mgen3 = GaussianObs(Y3, X3, params, lm3, is_out_positive=True)
            
                # Let's sample from X for later use.
#                 sampleY2, sampleX2 = mgen2.sample_XY(sess, Xvar_name='M2/X2:0', Nsamps=Nsamps,
#                                                      NTbins=50, with_inflow=True)
#                 sampleY3, sampleX3 = mgen3.sample_XY(sess, Xvar_name='M3/X3:0', Nsamps=Nsamps,
#                                                      NTbins=50, with_inflow=True)

#     def test_eval_Y_Gaussian(self):
#         """
#         TODO:
# 
#         Checks for the Gaussian mean and variance
#         """
#         print('X (mean, std, max)', np.mean(self.sampleX3), np.std(self.sampleX3),
#                   np.max(self.sampleX3))
#         print('Y (mean, std, max)', np.mean(self.sampleY3), np.std(self.sampleY3),
#                   np.max(self.sampleY3))
#         with self.sess.as_default():
#             SigmaChol = self.sess.run(self.mgen3.SigmaChol_NxTxDxD, feed_dict={'M3/X3:0' : self.sampleX3})
#             print('\nSigmaChol (mean, std)', np.mean(SigmaChol), np.std(SigmaChol))
#             mins, maxs = np.min(SigmaChol, axis=(0,1)).flatten(), np.max(SigmaChol, axis=(0,1)).flatten()
#             print('\nSigmaChol ranges')
#             for elem in list(zip(mins, maxs))[:10]: print(elem)
#             print('\nYpath')
#             for i, yvals in enumerate(self.sampleY3[0]): print(i, ',', yvals[:3])
#             print('\nXpath')
#             for i, xvals in enumerate(self.sampleX3[0]): print(i, ',', xvals[:3])
#             print(''
    
#     def test_logdensity(self):
#         """
#         Computes the LogDensity and checks that its values are reasonable. In
#         particular, for a particular generative model (`mgen1` above), it compares
#         the LD on external data with the LD on data it itself generates. The
#         latter should be smaller than the former.
#         """
#         with self.sess.as_default():
#             ld1_val = self.sess.run(self.ld1, feed_dict={'M1/X1:0' : self.sampleX1,
#                                                     'M1/Y1:0' : self.sampleY1})
#             cks1_val = self.sess.run(self.checks1, feed_dict={'M1/X1:0' : self.sampleX1,
#                                                     'M1/Y1:0' : self.sampleY1})
#             ld1_val2 = self.sess.run(self.ld1, feed_dict={'M1/X1:0' : self.sampleX2,
#                                                      'M1/Y1:0' : self.sampleY2})
#             cks2_val = self.sess.run(self.checks1, feed_dict={'M1/X1:0' : self.sampleX2,
#                                                     'M1/Y1:0' : self.sampleY2})
#             print('LogD << LogD with wrong data:', np.abs(ld1_val), '<<', abs(ld1_val2))
#             print('checks1', cks1_val)
#             print('checks2', cks2_val)
#             print('\n')
#             
    
#     def test_poisson_obvs(self):
#         """
#         Checks for the Poisson observations (integers).
#         """
#         with self.sess.as_default():
#             print('sampleX 1 (mean, std)', np.mean(self.sampleX1), np.std(self.sampleX1),
#                   np.max(self.sampleX1))
#             print('Ydata 1 (mean, std, max)', np.mean(self.sampleY1), np.std(self.sampleY1),
#                   np.max(self.sampleY1))
#             print('sampleX 2 (mean, std)', np.mean(self.sampleX2), np.std(self.sampleX2),
#                   np.max(self.sampleX2))
#             print('Ydata 2 (mean, std, max)', np.mean(self.sampleY2), np.std(self.sampleY2),
#                   np.max(self.sampleY2))
#             print('\nYpath')
#             for yvals in self.sampleY1[0]: print(yvals[:3])
# 
#             print('\n')
             
#     def test_eval_Y_and_rates_ispositive(self):
#         """
#         Checks for the Poisson rates with a softplus final layer in the
#         generating NN that yields positive numbers directly.
#         """
#         with self.sess.as_default():
#             rate2_NTxD = self.mgen2.rate_NTxD
#             sample_rate2 = self.sess.run(rate2_NTxD, feed_dict={'M2/X2:0' : self.sampleX2})        
#             print('Rate 2 (mean, std, max)', np.mean(sample_rate2), np.std(sample_rate2),
#                   np.max(sample_rate2))
#             print('')
#             
#     def test_eval_Y_and_rates(self):
#         """
#         Checks for the Poisson rates with a linear final layer in the generating
#         NN that needs to be further composed with an exponential activation to
#         obtain positive rates
#         """
#         with self.sess.as_default():
#             rate1_NTxD = self.mgen1.rate_NTxD            
#             sample_rate1 = self.sess.run(rate1_NTxD, feed_dict={'M1/X1:0' : self.sampleX1})
#             print('Rate 1 (mean, std, max)', np.mean(sample_rate1), np.std(sample_rate1),
#                   np.max(sample_rate1))
#             print('')
#     """
    
if __name__ == '__main__':
    tf.test.main()

