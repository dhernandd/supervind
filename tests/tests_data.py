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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf

from code.Optimizer_VAEC import Optimizer_TS

DTYPE = tf.float32
DATA_FILE = '/Users/danielhernandez/work/supervind/data/ziqiang/datadict'
# DATA_FILE = '/Users/danielhernandez/work/supervind/data/poisson_data_002/datadict'
# DATA_FILE = '/Users/danielhernandez/work/supervind/data/gaussian001/datadict'

# For information on these parameters, see runner.py
flags = tf.app.flags
flags.DEFINE_string('gen_mod_class', 'Gaussian', "")
flags.DEFINE_string('lat_mod_class', 'llinear', "")
flags.DEFINE_string('rec_mod_class', 'SmoothLl', "")
flags.DEFINE_integer('yDim', 18, "")
flags.DEFINE_integer('xDim', 5, "")
flags.DEFINE_float('alpha', 0.3, "")
flags.DEFINE_float('initrange_MuX', 5.0, "")
flags.DEFINE_float('initrange_LambdaX', 1.0,"")
flags.DEFINE_float('initrange_B', 1.0, "")
flags.DEFINE_float('initrange_outY', 3.0,"")
flags.DEFINE_float('init_Q0', 0.5, "")
flags.DEFINE_float('init_Q', 0.4, "")
flags.DEFINE_float('initrange_Goutmean', 0.3, "")
flags.DEFINE_float('initrange_Goutvar', 1.0, "")
flags.DEFINE_float('initbias_Goutmean', 1.0, "")
flags.DEFINE_float('learning_rate', 2e-3, "")
flags.DEFINE_boolean('use_grad_term', True, "")
params = tf.flags.FLAGS


class DataTests(tf.test.TestCase):
    """
    """
    with open(DATA_FILE, 'rb+') as f:
        datadict = pickle.load(f, encoding='latin1') # `encoding='latin1'` for python 2 pickled data 
        Ydata = 2*datadict['Ytrain']
        Yvalid = 2*datadict['Yvalid']
        
    yDim = Ydata.shape[2]
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            opt = Optimizer_TS(params)
            mrec = opt.mrec
            mlat = opt.lat_ev_model
            mgen = opt.mgen
            
            sess.run(tf.global_variables_initializer())
            
    def test_logdensity(self):
        sess = self.sess
        with sess.as_default():
            MuX = sess.run(self.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata})
            cost = sess.run(self.opt.cost, feed_dict={'VAEC/Y:0' : self.Ydata,
                                                      'VAEC/X:0' : MuX})
            print('cost:', cost)
            postX = sess.run(self.mrec.postX_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata,
                                                         'VAEC/X:0' : MuX})
            cost = sess.run(self.opt.cost, feed_dict={'VAEC/Y:0' : self.Ydata,
                                                      'VAEC/X:0' : postX})
            checks = sess.run(self.opt.checks1, feed_dict={'VAEC/Y:0' : self.Ydata,
                                                          'VAEC/X:0' : postX})
            print('cost:', cost)
            print('checks1', checks)
            
    def test_postX(self):
        if params.gen_mod_class == 'Gaussian':
            sess = self.sess
            with sess.as_default():
                MuX = sess.run(self.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata})
                MuX_valid = sess.run(self.mrec.Mu_NxTxd, feed_dict={'VAEC/Y:0' : self.Yvalid})
                postX = sess.run(self.mrec.postX_NxTxd, feed_dict={'VAEC/Y:0' : self.Ydata,
                                                             'VAEC/X:0' : MuX})
                postX_valid = sess.run(self.mrec.postX_NxTxd, feed_dict={'VAEC/Y:0' : self.Yvalid,
                                                                   'VAEC/X:0' : MuX_valid})
                Yprime = sess.run(self.opt.mgen.MuY_NxTxD, feed_dict={'VAEC/X:0' : postX})
                SigmaY = sess.run(self.mgen.SigmaInvY_DxD)
                
                cost = sess.run(self.opt.cost, feed_dict={'VAEC/X:0' : MuX,
                                                          'VAEC/Y:0' : self.Ydata})
                checks = sess.run(self.opt.checks1, feed_dict={'VAEC/X:0' : MuX,
                                                              'VAEC/Y:0' : self.Ydata})
                new_valid_cost = sess.run(self.opt.cost, feed_dict={'VAEC/X:0' : MuX_valid,
                                                                'VAEC/Y:0' : self.Yvalid})
                checks_v = sess.run(self.opt.checks1, feed_dict={'VAEC/X:0' : MuX_valid,
                                                              'VAEC/Y:0' : self.Yvalid})
    #             checks2 = sess.run(self.mrec.checks1, feed_dict={'VAEC/X:0' : MuX_valid,
    #                                                           'VAEC/Y:0' : self.Yvalid})
                
                print('Yprime (mean, std)', np.mean(Yprime), np.std(Yprime))
                mins, maxs = np.min(Yprime, axis=(0,1)), np.max(Yprime, axis=(0,1))
                print('Yprime ranges', list(zip(mins, maxs)))
                print('\nSigmaY', SigmaY[0,0])
                print('SigmaY', np.linalg.det(SigmaY))
                
                print('\npostX (mean, std)', np.mean(postX), np.std(postX))
                
                mins, maxs = np.min(postX, axis=(0,1)), np.max(postX, axis=(0,1))
                print('postX ranges', list(zip(mins, maxs)))
                print('postX_valid (mean, std)', np.mean(postX_valid), np.std(postX_valid))
                mins, maxs = np.min(postX_valid, axis=(0,1)), np.max(postX_valid, axis=(0,1))
                print('postX_valid ranges', list(zip(mins, maxs)))
                
                print('cost', cost, checks)
                print('valid cost', new_valid_cost, checks_v)
                print('')

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