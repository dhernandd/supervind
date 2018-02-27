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
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf

from code.Optimizer_VAEC import Optimizer_TS

DTYPE = tf.float32
DATA_FILE = '/Users/danielhernandez/work/vind/data/poisson_data_002/datadict'
# DATA_FILE = '/Users/danielhernandez/work/supervind/data/poisson002/datadict'

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_integer('yDim', 10, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 0.5, "")
flags.DEFINE_float('init_Q', 2.0, "")
flags.DEFINE_float('alpha', 0.5, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
params = tf.flags.FLAGS


class DataTests(tf.test.TestCase):
    """
    """
    with open(DATA_FILE, 'rb+') as f:
        datadict = pickle.load(f, encoding='latin1') # `encoding='latin1'` for python 2 pickled data 
        Ydata = datadict['Ytrain']
        
    yDim = Ydata.shape[2]
    xDim = 2
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            opt = Optimizer_TS(params)
            mrec = opt.mrec
            mlat = opt.lat_ev_model
            mgen = opt.mgen
            
            sess.run(tf.global_variables_initializer())
            
    def test_data(self):
        print('Y (mean, std):', np.mean(self.Ydata), np.std(self.Ydata))
        print('Y range:', np.min(self.Ydata), np.max(self.Ydata))
        print('')
            
    def test_inferredX_range(self):
        """
        This computes the initial ranges for the values of the latent-space
        variables inferred by the recognition network for your data. Reasonable
        values per dimension are -30 <~ min(MuX) <~ max(MuX) < 30. 
        """
        sess = self.sess
        with sess.as_default():
            MuX = sess.run(self.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata})
            print('MuX (mean, std)', np.mean(MuX), np.std(MuX))
            mins, maxs = np.min(MuX, axis=(0,1)), np.max(MuX, axis=(0,1))
            print('MuX ranges', list(zip(mins, maxs)))
            print('')
    
    def test_inferredLambdaX_range(self):
        """
        This computes the initial ranges for the values of the latent-space
        precision as yielded by the recognition network for your data. Reasonable
        values per LambdaX entry L are -3 < min(L) < max(L) < 3
        """
        sess = self.sess
        with sess.as_default():
            LambdaX = sess.run(self.mrec.Lambda_NxTxdxd, feed_dict={'VAEC/Y:0' : self.Ydata})
            print('LambdaX (mean, std)', np.mean(LambdaX), np.std(LambdaX))
            mins, maxs = np.min(LambdaX, axis=(0,1)).flatten(), np.max(LambdaX, axis=(0,1)).flatten()
            print('LambdaX ranges', list(zip(mins, maxs)))
            print('')
        
    def test_nonlinearity_range(self):
        """
        The average and max values of the nonlinearity alpha*B should be <~
        o(10^-1). That is smaller than 1, yet sizable. This depends on the
        nonlinearity network and on the range of the values in the latent space
        that the recognition network yields.
        """
        sess = self.sess
        with sess.as_default():
            MuX = sess.run(self.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata})
            alphaB = sess.run(self.mlat.alpha*self.mlat.B_NxTxdxd,
                              feed_dict={'VAEC/X:0' : MuX})
            print('alphaB (mean, std)', np.mean(alphaB), np.std(alphaB))
            mins, maxs = np.min(alphaB, axis=(0,1)).flatten(), np.max(alphaB, axis=(0,1)).flatten()
            print('alphaB ranges', list(zip(mins, maxs)))
            
        
if __name__ == '__main__':

    tf.test.main()