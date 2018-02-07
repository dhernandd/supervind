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


from Optimizer_VAEC import Optimizer_TS

# __package__ = 'supervind.tests'


class OptimizerTest(tf.test.TestCase):
    """
    """
    xDim = 2
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.3, -1.4], [6.7, 8.9]]])
    opt = Optimizer_TS(xDim)
    
    def test_simple(self):
        print('xDim:', self.opt.xDim)
        with tf.Session(graph=self.opt.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Nsamps = sess.run(self.opt.lat_ev_model.Nsamps, feed_dict={'X:0' : self.Xdata1})
            print('Nsamps', Nsamps)
        
        
if __name__ == '__main__':

    tf.test.main()