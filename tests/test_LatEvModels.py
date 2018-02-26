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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from code.LatEvModels import LocallyLinearEvolution

DTYPE = tf.float32

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_integer('yDim', 50, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 1.0, "")
flags.DEFINE_float('init_Q', 2.0, "")
flags.DEFINE_float('alpha', 0.5, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
params = tf.flags.FLAGS

class LocallyLinearEvolutionTest(tf.test.TestCase):
    """
    The algorithm requires to repeatedly add tensors A to the graph following a rule:
    
    A = f(X)

    where the input X varies but f does not. The main purpose of these tests is
    to check that the implementation of this aspect is working as desired
    """
    xDim = 2
    
    # For some reason unbeknownst to man, tensorflow has decided to set in stone
    # the numpy RandomState in its test classes. At least that is what I gather
    # from many tests after getting the same result over and over in experiments
    # that should have been in principle independent. Obviously, this is
    # undocumented, as it is undocumented how to change the seed to random. I
    # mean, why would anyone feel the need to document that?! Breathe. Hack
    # follows.
    seed_list = np.random.randint(1000, size=100).tolist()
    Nsamps = 100
    NTbins = 50
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('LM1'):
                X1 = tf.placeholder(DTYPE, [None, None, xDim], 'X1')
                lm1 = LocallyLinearEvolution(X1, params)
                LD1_winflow, _ = lm1.compute_LogDensity_Xterms(X1, with_inflow=True) 
            with tf.variable_scope('LM2'):
                X2 = tf.placeholder(DTYPE, [None, None, xDim], 'X2')
                lm2 = LocallyLinearEvolution(X2, params)
                LD2_winflow, _ = lm2.compute_LogDensity_Xterms(X2, with_inflow=True) 
            
            # Let's sample from X for later use.
            sess.run(tf.global_variables_initializer())
            sampleX1 = lm1.sample_X(sess, 'LM1/X1:0', with_inflow=True, Nsamps=Nsamps, NTbins=NTbins,
                                    draw_plots=True, init_variables=False)
            sampleX2 = lm2.sample_X(sess, 'LM2/X2:0', with_inflow=True, Nsamps=Nsamps, NTbins=NTbins,
                                    draw_plots=False, init_variables=False)
    
    def test_LogDensity(self):
        """
        Evaluates the cost from the latent evolution instance. 
        """
        Nsamps = self.Nsamps
        with self.sess.as_default():
            LD1, arrays = self.lm1.logdensity_Xterms
            LD1_vals = self.sess.run([LD1, self.LD1_winflow],
                                     feed_dict={'LM1/X1:0' : self.sampleX1})
            L = self.sess.run([arrays[0], arrays[1],arrays[2], arrays[3], arrays[4]],
                              feed_dict={'LM1/X1:0' : self.sampleX1})
            
            LD2, _ = self.lm2.logdensity_Xterms
            LD2_vals = self.sess.run([LD2, self.LD2_winflow],
                                    feed_dict={'LM2/X2:0' : self.sampleX1})
            
 
            print('LD1 [wout, w] inflow:', LD1_vals[0]/Nsamps, LD1_vals[1]/Nsamps)
            print('LD2 on X1 data [wout, w] inflow:', LD2_vals[0]/Nsamps, LD2_vals[1]/Nsamps)
            self.assertGreater(np.abs(LD2_vals[1]), np.abs(LD1_vals[1]))
            print('[L0, L1, L2, L3, L4]:', np.array(L)/Nsamps)
            print('\n')
 
    def test_Bstatistics(self):
        """
        Compute B statistics. This is a control check to see that the
        nonlinearity is reasonably small. Mean B should be around 0 with stddev
        around 0.1.
        """
        with self.sess.as_default():
            alphaB_NxTxdxd = self.sess.run(self.lm1.alpha*self.lm1.B_NxTxdxd,
                                           feed_dict={'LM1/X1:0' : self.sampleX1})
            print('alphaB (mean, std):', np.mean(alphaB_NxTxdxd), np.std(alphaB_NxTxdxd))
            mins, maxs = np.min(alphaB_NxTxdxd, axis=(0,1)).flatten(), np.max(alphaB_NxTxdxd,
                                                                              axis=(0,1)).flatten()
            print('alphaB ranges', list(zip(mins, maxs)))
            print('\n')
             

if __name__ == '__main__':
    tf.test.main()



