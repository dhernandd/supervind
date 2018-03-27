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

from code.ObservationModels import CellVoltageObs
from code.RecogLLEv_wParams import SmoothingNLDSTimeSeries

DTYPE = tf.float32

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_string('lat_mod_class', 'llinear', "")
flags.DEFINE_integer('yDim', 1, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_integer('pDim', 1, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_LambdaX', 1.0, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 1.0, "")
flags.DEFINE_float('init_Q', 1.0, "")
flags.DEFINE_float('alpha', 0.5, "")
flags.DEFINE_float('initrange_outY', 1.0,"")
flags.DEFINE_float('initrange_Goutmean', 0.03,"")
flags.DEFINE_float('initrange_Goutvar', 2.0,"")
flags.DEFINE_float('initbias_Goutmean', 1.0,"")
flags.DEFINE_integer('genNsamps', 100, "")
flags.DEFINE_integer('genNTbins', 30, "")
flags.DEFINE_integer('num_diff_entities', 2, "")
flags.DEFINE_boolean('use_grads', False, "")
params = tf.flags.FLAGS


class SmoothingNLDSTimeSeriesTest(tf.test.TestCase):
    """
    """
    xDim = params.xDim
    NTbins = params.genNTbins
    Nsamps = params.genNsamps
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('M1'):
                X1 = tf.placeholder(DTYPE, [1, None, xDim], 'X1')
                Y1 = tf.placeholder(DTYPE, [1, None, 1], 'Y1')
                mrec1 = SmoothingNLDSTimeSeries(Y1, X1, params) 
                lm1 = mrec1.get_lat_ev_model()
                LD1_winflow, _ = lm1.compute_LogDensity_Xterms(X1, with_inflow=True) 
                
                mgen1 = CellVoltageObs(Y1, X1, params, lm1, is_out_positive=True)
                ld1, checks1 = mgen1.compute_LogDensity(with_inflow=True)
            
                # Let's sample from X, Y.
                sess.run(tf.global_variables_initializer())
                sampleY1, sampleX1, Ids = mgen1.sample_XY(sess, feed_key='M1/X1:0', Nsamps=Nsamps,
                                                     NTbins=NTbins, with_inflow=True)
#             mean, std = np.mean(sampleX1, axis=(0,1)), np.std(sampleX1, axis=(0,1))
#             print('sampleX (mean, std, max)', list(zip(mean, std)))
#             mins, maxs = np.min(sampleX1, axis=(0,1)), np.max(sampleX1, axis=(0,1))
#             print('sampleX ranges', list(zip(mins, maxs)))
#             print("")
#             
#             print('Y (mean, std, max)', np.mean(sampleY1), np.std(sampleY1),
#                       np.max(sampleY1), '\n')
            
#     def test_Entropy(self):
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             Entropy = sess.run(self.mrec1.Entropy, feed_dict={'M1/Y1:0' : self.sampleY1,
#                                                               'M1/X1:0' : self.sampleX1})
#             print('Entropy:', Entropy)
#             print('')
# 
#     def test_postX(self):
#         with tf.Session(graph=self.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             postX = sess.run(self.mrec1.postX, feed_dict={'M1/Y1:0' : self.sampleY1,
#                                                           'M1/X1:0' : self.sampleX1})
#             mean, std = np.mean(postX, axis=(0,1)), np.std(postX, axis=(0,1))
#             print('postX (mean, std, max)', list(zip(mean, std)))
#             mins, maxs = np.min(postX, axis=(0,1)), np.max(postX, axis=(0,1))
#             print('postX ranges', list(zip(mins, maxs)))
#             print("")
#             
#     def test_MuLambda_statistics(self):
#         with self.sess.as_default():
#             Mu = self.sess.run(self.mrec1.Mu_NxTxd, feed_dict={'M1/Y1:0' : self.sampleY1})
#             Lambda = self.sess.run(self.mrec1.Lambda_NxTxdxd, feed_dict={'M1/Y1:0' : self.sampleY1})
#             LambdaMu = self.sess.run(self.mrec1.LambdaMu_NxTxd, feed_dict={'M1/Y1:0' : self.sampleY1})
#             print('Mu (mean, rate, max):', np.mean(Mu), np.std(Mu), np.max(Mu))
#             print('Lambda (mean, rate, max)', np.mean(Lambda), np.std(Lambda), np.max(Lambda))
#             print('LambdaMu:', np.mean(LambdaMu), np.std(LambdaMu), np.max(LambdaMu))
#             print('\n')
             
 
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
