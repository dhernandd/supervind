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
import os
import pickle

import matplotlib
matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from code.Optimizer_VAEC import Optimizer_TS
from code.datetools import addDateTime

# DATA_FILE = '/Users/danielhernandez/work/supervind/data/poisson002/datadict'
DATA_FILE = '/Users/danielhernandez/work/vind/data/poisson_data_002/datadict'
RLT_DIR = "/Users/danielhernandez/work/supervind/rslts/poisson002/fit" + addDateTime()+'/'
if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)


# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_integer('yDim', 10, "")
flags.DEFINE_integer('xDim', 2, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_float('initrange_MuX', 0.2, "")
flags.DEFINE_float('initrange_B', 3.0, "")
flags.DEFINE_float('init_Q0', 0.4, "")
flags.DEFINE_float('init_Q', 1.0, "")
flags.DEFINE_float('alpha', 0.2, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
flags.DEFINE_float('initrange_LambdaX', 1.0, "")
params = tf.flags.FLAGS


def write_option_file(path):
    """
    """
    params_list = sorted([param for param in dir(params) if param 
                          not in ['h', 'help', 'helpfull', 'helpshort']])
    with open(path + 'params.txt', 'w') as option_file:
        for par in params_list:
            option_file.write(par + ' ' + str(getattr(params, par)) + '\n')


class OptimizerTest(tf.test.TestCase):
    """
    """
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            opt = Optimizer_TS(params)
            sess.run(tf.global_variables_initializer())

    def test_train(self):
        sess = self.sess
        with open(DATA_FILE, 'rb+') as f:
#             datadict = pickle.load(f, encoding='latin1')
            datadict = pickle.load(f, encoding='latin1')
            Ydata = datadict['Ytrain']
            Yvalid = datadict['Yvalid']
        with sess.as_default():
            X = sess.run(self.opt.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : Ydata})
            Xv = sess.run(self.opt.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : Yvalid})
            print(len(X), len(Xv))
            pX = sess.run(self.opt.mrec.postX, feed_dict={'VAEC/Y:0' : Ydata,
                                                          'VAEC/X:0' : X})
            pXv = sess.run(self.opt.mrec.postX, feed_dict={'VAEC/Y:0' : Yvalid,
                                                          'VAEC/X:0' : Xv})
            print(len(pX), len(pXv))
            new_valid_cost = sess.run(self.opt.cost, feed_dict={'VAEC/X:0' : Xv,
                                                            'VAEC/Y:0' : Yvalid})
            print(new_valid_cost)
            self.opt.train(sess, RLT_DIR, Ydata)
            print(sess.run(self.opt.cost, feed_dict={'VAEC/X:0' : Xv,
                                                     'VAEC/Y:0' : Yvalid}))

        
        
if __name__ == '__main__':
    write_option_file(RLT_DIR)
    tf.test.main()