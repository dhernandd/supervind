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

from .ObservationModels import PoissonObs, GaussianObs
from .RecognitionModels import SmoothingNLDSTimeSeries
from .datetools import addDateTime

DTYPE = tf.float32

def data_iterator_simple(Ydata, Xdata, Ids=None, batch_size=1, shuffle=True):
    """
    """
    l_inds = np.arange(len(Xdata))
    if shuffle: 
        np.random.shuffle(l_inds)
    if Ids is None:
        for i in range(0, len(Ydata), batch_size):
            yield Ydata[l_inds[i:i+batch_size]], Xdata[l_inds[i:i+batch_size]]
    else:
        for i in range(0, len(Ydata), batch_size):
            yield ( Ydata[l_inds[i:i+batch_size]], Xdata[l_inds[i:i+batch_size]],
                    Ids[l_inds[i:i+batch_size]] )


class Optimizer_TS():
    """
    """
    def __init__(self, params):
        """
        """        
        self.params = params

        gen_mod_classes = {'Poisson' : PoissonObs, 'Gaussian' : GaussianObs}
        rec_mod_classes = {'SmoothLl' : SmoothingNLDSTimeSeries}

        ObsModel = gen_mod_classes[params.gen_mod_class]
        RecModel = rec_mod_classes[params.rec_mod_class]

        self.xDim = xDim = params.xDim
        self.yDim = yDim = params.yDim
        
        with tf.variable_scope('VAEC', reuse=tf.AUTO_REUSE):
            self.learning_rate = lr = tf.get_variable('lr', dtype=DTYPE,
                                                      initializer=params.learning_rate)
            self.Y = Y = tf.placeholder(DTYPE, [None, None, yDim], name='Y')
            self.X = X = tf.placeholder(DTYPE, [None, None, xDim], name='X')
            self.mrec = RecModel(Y, X, params)
#             
            self.lat_ev_model = lat_ev_model = self.mrec.lat_ev_model
            self.mgen = ObsModel(Y, X, params, lat_ev_model)
            
            self.cost_ng, self.checks1 = self.cost_ELBO()
            self.cost, self.checks2 = self.cost_ELBO(use_grads=True)
            self.cost_with_inflow, _ = self.cost_ELBO(with_inflow=True)
            
            self.ELBO_summ = tf.summary.scalar('ELBO', self.cost_ng)
            
            # The optimizer ops
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope=tf.get_variable_scope().name)
            print('Scope', tf.get_variable_scope().name)
            for i in range(len(self.train_vars)):
                shape = self.train_vars[i].get_shape().as_list()
                print("    ", i, self.train_vars[i].name, shape)
            
            opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
            
            self.gradsvars_ng = gradsvars_ng = opt.compute_gradients(self.cost_ng, self.train_vars)
            self.gradsvars = gradsvars = opt.compute_gradients(self.cost, self.train_vars)
            self.train_step = tf.get_variable("global_step", [], tf.int64,
                                              tf.zeros_initializer(),
                                              trainable=False)
            self.train_op_ng = opt.apply_gradients(gradsvars_ng, global_step=self.train_step)
            self.train_op = opt.apply_gradients(gradsvars, global_step=self.train_step)
            
            self.saver = tf.train.Saver(tf.global_variables())


    def cost_ELBO(self, with_inflow=False, use_grads=False):
        """
        """
        noisy_postX = self.mrec.noisy_postX if use_grads else self.mrec.noisy_postX_ng
        LogDensity, LDchecks = self.mgen.compute_LogDensity(noisy_postX, with_inflow=with_inflow) # checks=[LX0, LX1, LX2, LX3, LX4, LX, LY, LY1, LY2]
        Entropy = self.mrec.compute_Entropy(noisy_postX)
        
        checks = [LogDensity, Entropy]
        checks.extend(LDchecks)
        
        return -(LogDensity + Entropy), checks 

    def train(self, sess, rlt_dir, Ytrain, Yvalid=None, Ids_train=None, Ids_valid=None,
              num_epochs=2000):
        """
        Initialize all variables outside this method.
        """
        params = self.params
        
        Ytrain_NxTxD = Ytrain
        Nsamps = len(Ytrain)
        if Yvalid is not None:
            Yvalid_VxTxD, with_valid = Yvalid, True
            Nsamps_valid = len(Yvalid_VxTxD)
        else: with_valid = False
        if Ids_train is None:
            Ids_train = np.zeros(shape=[Nsamps], dtype=np.int32)
            Ids_valid = np.zeros(shape=[Nsamps_valid], dtype=np.int32)
        started_training = False
        
        # Placeholder for some more summaries that may be of interest.
        LD_summ = tf.summary.scalar('LogDensity', self.checks1[0])
        E_summ = tf.summary.scalar('Entropy', self.checks1[1])
        LY_summ = tf.summary.scalar('LY', self.checks1[2])
        LX_summ = tf.summary.scalar('LX', self.checks1[3])
        merged_summaries = tf.summary.merge([LD_summ, E_summ, LY_summ, LX_summ, self.ELBO_summ])

        self.writer = tf.summary.FileWriter(addDateTime('./logs/log'))
        valid_cost = np.inf
        for ep in range(num_epochs):
            if self.params.use_grad_term:
                postX = self.mrec.postX_NxTxd
            else:
                if ep > params.num_eps_to_include_grads:
                    print("Including the grad term from now on...")
                    self.params.use_grad_term = True
                    postX = self.mrec.postX_NxTxd
                else:
                    postX = self.mrec.postX_ng_NxTxd

            # The Fixed Point Iteration step. This is the key to the
            # algorithm.
            if not started_training:
                Xpassed_NxTxd = sess.run(self.mrec.Mu_NxTxd, 
                                         feed_dict={'VAEC/Y:0' : Ytrain_NxTxD}) 
                started_training = True
                if with_valid:
                    Xvalid_VxTxd = sess.run(self.mrec.Mu_NxTxd,
                                            feed_dict={'VAEC/Y:0' : Yvalid_VxTxD})
            else:
                for _ in range(self.params.num_fpis):
                    Xpassed_NxTxd = sess.run(postX, feed_dict={'VAEC/Y:0' : Ytrain_NxTxD,
                                                               'VAEC/X:0' : Xpassed_NxTxd,
                                                               'VAEC/Ids:0' : Ids_train})
                if with_valid:
                    Xvalid_VxTxd = sess.run(postX, feed_dict={'VAEC/Y:0' : Yvalid_VxTxD,
                                                              'VAEC/X:0' : Xvalid_VxTxd,
                                                              'VAEC/Ids:0' : Ids_valid})
            
            # The gradient descent step
            lr = params.learning_rate - ep/num_epochs*(params.learning_rate - params.end_lr)
            train_op = self.train_op if self.params.use_grad_term else self.train_op_ng
            iterator_YX = data_iterator_simple(Ytrain_NxTxD, Xpassed_NxTxd, Ids_train,
                                               batch_size=params.batch_size,
                                               shuffle=params.shuffle)
            for batch_y, batch_x, batch_id in iterator_YX:
                _ = sess.run([train_op], feed_dict={'VAEC/X:0' : batch_x,
                                                    'VAEC/Y:0' : batch_y,
                                                    'VAEC/lr:0' : lr,
                                                    'VAEC/Ids:0' : batch_id})

            cost, summaries = sess.run([self.cost, merged_summaries], 
                                       feed_dict={'VAEC/X:0' : Xpassed_NxTxd,
                                                  'VAEC/Y:0' : Ytrain_NxTxD,
                                                  'VAEC/Ids:0' : Ids_train})
            self.writer.add_summary(summaries, ep)
            print('Ep, Cost:', ep, cost/Nsamps)

            # Save if validation improvement
            if ep % 10 == 0:
                if self.xDim == 2:
                    if params.with_ids:
                        for ent in range(self.params.num_diff_entities):
                            print('Plottins DS for entity ', str(ent), '...')
                            list_idxs = [i for i, Id in enumerate(Ids_train) if Id == ent]
                            print(list_idxs)
                            Xdata_thisid = Xpassed_NxTxd[list_idxs]
                            self.lat_ev_model.plot_2Dquiver_paths(sess, Xdata_thisid, 'VAEC/X:0', 
                                                                  rlt_dir=rlt_dir,
                                                                  rslt_file='qplot'+str(ep)+'_'+str(ent),
                                                                  savefig=True, draw=False, skipped=5,
                                                                  feed_range=False, Id=ent)
                    else:
                        self.lat_ev_model.plot_2Dquiver_paths(sess, Xpassed_NxTxd, 'VAEC/X:0', 
                                                              rlt_dir=rlt_dir, rslt_file='qplot'+str(ep),
                                                              savefig=True, draw=False, skipped=5)
                if with_valid:
                    new_valid_cost = sess.run(self.cost, feed_dict={'VAEC/X:0' : Xvalid_VxTxd,
                                                                    'VAEC/Y:0' : Yvalid_VxTxD,
                                                                    'VAEC/Ids:0' : Ids_valid})
                    if new_valid_cost < valid_cost:
                        valid_cost = new_valid_cost
                        print('Valid. cost:', valid_cost, '... Saving...')
                        self.saver.save(sess, rlt_dir+'vaec', global_step=self.train_step)
            print('')


