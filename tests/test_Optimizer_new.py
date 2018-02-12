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
    yDim = 10
    xDim = 2
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.3, -1.4], [6.7, 8.9]]])

    opt = Optimizer_TS(yDim, xDim)
<<<<<<< HEAD
    
<<<<<<< HEAD
=======
>>>>>>> 4c5c36502666a34bd96835f1269d133585771520
#     with tf.Session(graph=opt.graph) as sess:
#         sess.run(tf.global_variables_initializer())
#         print('Generating some data...')
#         Ydata, Xdata = opt.mgen.sample_XY(sess, init_variables=False,
#                                            with_inflow=True, Nsamps=1)
#         print('Done')
=======
    with tf.Session(graph=opt.graph) as sess:
        sess.run(tf.global_variables_initializer())
        print('Generating some data...')
        Ydata, Xdata = opt.mgen.sample_XY(sess, init_variables=False,
                                           with_inflow=True, Nsamps=50,
                                           feed_key='VAEC/X:0')
#         print(Ydata, Xdata)
>>>>>>> develop
        
    def test_simple(self):
        print('xDim:', self.opt.xDim)
        with tf.Session(graph=self.opt.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Nsamps = sess.run(self.opt.lat_ev_model.Nsamps, 
                              feed_dict={'VAEC/X:0' : self.Xdata1})
#             print('Nsamps:', Nsamps)
#             print('Inv tau:', self.opt.mgen.inv_tau)

    
    def test_cost(self):
        with tf.Session(graph=self.opt.graph) as sess:
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = self.opt.mgen.sample_XY(sess, init_variables=False,
                                              with_inflow=True, Nsamps=1,
                                              feed_key='VAEC/X:0')
            cost = self.opt.cost_with_inflow
            cost_val1 = sess.run(cost, feed_dict={'VAEC/X:0' : Xdata,
                                                 'VAEC/Y:0' : Ydata})
            cost_val2 = sess.run(cost, feed_dict={'VAEC/X:0' : self.Xdata,
                                                 'VAEC/Y:0' : self.Ydata})
            print('Cost on self-generated data:', cost_val1)
            print('Cost on external data:', cost_val2)


    def test_train(self):
        self.opt.train(self.Ydata)

#     def test_train_op(self):
#         with tf.Session(graph=self.opt.graph) as sess:
#             sess.run(tf.global_variables_initializer())
#             Ydata, Xdata = self.opt.mgen.sample_XY(sess, init_variables=False,
#                                               with_inflow=True, Nsamps=1,
#                                               feed_key='VAEC/X:0')
#             train_op = self.opt.train_op
#             train_op_val = sess.run(train_op, feed_dict={'VAEC/X:0' : Xdata,
#                                                          'VAEC/Y:0' : Ydata})
#             print(train_op_val)
        
            
        
        
if __name__ == '__main__':

    tf.test.main()