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

import numpy as np

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from code.LatEvModels import LocallyLinearEvolution
from code.ObservationModels import PoissonObs, GaussianObs
from code.Optimizer_VAEC import Optimizer_TS
from code.datetools import addDateTime

DTYPE = tf.float32

# CONFIGURATION
RUN_MODE = 'train' # ['train', 'generate']

# DIRECTORIES, SAVE FILES, ETC
LOCAL_ROOT = "/Users/danielhernandez/work/supervind/"
LOCAL_DATA_DIR = "/Users/danielhernandez/work/supervind/data/" 
THIS_DATA_DIR = 'gaussian001/'
LOCAL_RLT_DIR = "/Users/danielhernandez/work/supervind/rslts/"
LOAD_CKPT_DIR = ""  # TODO:
SAVE_DATA_FILE = "datadict"
SAVE_TO_VIND = False
IS_PY2 = True

# MODEL/OPTIMIZER ATTRIBUTES
LAT_MOD_CLASS = 'llinear'
GEN_MOD_CLASS = 'Gaussian' # ['Gaussian', 'Poisson']
YDIM = 20  # TODO: yDim should be detected from data on train mode
XDIM = 2
NNODES = 60
ALPHA = 0.3
INITRANGE_MUX = 1.0
INITRANGE_LAMBDAX = 0.5
INITRANGE_B = 3.0
INITRANGE_OUTY = 3.0
INIT_Q0 = 0.7
INIT_Q = 0.5
INITRANGE_GOUTMEAN = 0.03
INITRANGE_GOUTVAR = 1.0
INITBIAS_GOUTMEAN = 1.0

# TRAINING PARAMETERS
LEARNING_RATE = 1e-2

# GENERATION PARAMETERS
NTBINS = 30
NSAMPS = 100
DRAW_HEAT_MAPS = True

flags = tf.app.flags
flags.DEFINE_string('mode', RUN_MODE, "The mode in which to run. Can be ['train', 'generate']")

flags.DEFINE_string('local_root', LOCAL_ROOT, "The root directory of supervind.")
flags.DEFINE_string('local_data_dir', LOCAL_DATA_DIR, "The directory that stores all the datasets")
flags.DEFINE_string('local_rlt_dir', LOCAL_RLT_DIR, "The directory that stores all the results")
flags.DEFINE_string('this_data_dir', THIS_DATA_DIR, ("For the 'generate' mode, the directory that shall "
                                                     "store this dataset"))
flags.DEFINE_string('save_data_file', SAVE_DATA_FILE, ("For the 'generate' mode, the name of the file "
                                                       "to store the data"))
flags.DEFINE_string('load_data_file', LOAD_CKPT_DIR, ("For the 'train' mode, the directory storing "
                                                       "`tf` checkpoints."))
flags.DEFINE_boolean('save_to_vind', SAVE_TO_VIND, ("Should the data be saved in a format that can be " 
                                                    "read by the old theano code"))
flags.DEFINE_boolean('is_py2', IS_PY2, "Was the data pickled in python 2?")

flags.DEFINE_integer('xDim', XDIM, "The dimensionality of the latent space")
flags.DEFINE_integer('yDim', YDIM, "The dimensionality of the data")
flags.DEFINE_string('lat_mod_class', LAT_MOD_CLASS, ("The evolution model class. Implemented "
                                                     "['llinear']"))
flags.DEFINE_string('gen_mod_class', GEN_MOD_CLASS, ("The generative model class. Implemented "
                                                     "['Poisson, Gaussian']"))
flags.DEFINE_float('alpha', ALPHA, ("The scale factor of the nonlinearity. This parameters "
                                    "works in conjunction with initrange_B"))
flags.DEFINE_float('initrange_MuX', INITRANGE_MUX, ("Controls the initial ranges within "
                                           "which the latent space paths are contained. Bigger "
                                           "values here lead to bigger bounding box. It is im-"
                                           "portant to adjust this parameter so that the initial "
                                           "paths do not collapse nor blow up."))
flags.DEFINE_float('initrange_LambdaX', INITRANGE_LAMBDAX, ("Controls the initial ranges within "
                                                "which the latent space paths are contained. Roughly "
                                                "rangeX ~ 1/(Lambda + Q), so if Lambda very big, the "
                                                "range is reduced. If Lambda very small, then it defers "
                                                "to Q. Optimally Lambda ~ Q ~ 1."))
flags.DEFINE_float('initrange_B', INITRANGE_B, ("Controls the initial size of the nonlinearity. "
                                                "Works in conjunction with alpha"))
flags.DEFINE_float('initrange_outY', INITRANGE_OUTY, ("Controls the initial range of the output of the "
                                                "generative network"))
flags.DEFINE_float('init_Q0', INIT_Q0, ("Controls the initial spread of the starting points of the "
                                    "paths in latent space."))
flags.DEFINE_float('init_Q', INIT_Q, ("Controls the initial noise added to the paths in latent space. "
                                      "More importantly, it also controls the initial ranges within "
                                      "which the latent space paths are contained. Roughly rangeX ~  "
                                      "1/(Lambda + Q), so if Q is very big, the range is reduced. If "
                                      "Q is very small, then it defers to Lambda. Optimally "
                                      "Lambda ~ Q ~ 1."))
flags.DEFINE_float('initrange_Goutmean', INITRANGE_GOUTMEAN, "")
flags.DEFINE_float('initrange_Goutvar', INITRANGE_GOUTVAR, "")
flags.DEFINE_float('initbias_Goutmean', INITBIAS_GOUTMEAN, "")


flags.DEFINE_float('learning_rate', LEARNING_RATE, "It's the learning rate, silly")

flags.DEFINE_integer('genNsamps', NSAMPS, "The number of samples to generate")
flags.DEFINE_integer('genNTbins', NTBINS, "The number of time bins in the generated data")
flags.DEFINE_boolean('draw_heat_maps', DRAW_HEAT_MAPS, "Should I draw heat maps of your data?")

params = tf.flags.FLAGS


def write_option_file(path):
    """
    Writes a file with the parameters that were used for this fit. Cuz you will
    forget.
    """
    params_list = sorted([param for param in dir(params) if param 
                          not in ['h', 'help', 'helpfull', 'helpshort']])
    with open(path + 'params.txt', 'w') as option_file:
        for par in params_list:
            option_file.write(par + ' ' + str(getattr(params, par)) + '\n')
                
def generate_fake_data(lat_mod_class, gen_mod_class, params,
                       data_path=None,
                       save_data_file=None,
                       Nsamps=100,
                       NTbins=30,
                       write_params_file=False,
                       draw_quiver=False,
                       draw_heat_maps=True,
                       savefigs=False):
    """
    Generates synthetic data and possibly pickles it for later use. Maybe you
    would like to train a model? 
    
    Args:
        lat_mod_class: A string that is a key to the evolution model class. Currently 
                    'llinear' -> `LocallyLinearEvolution` is implemented.
        gen_mod_class: A string that is a key to the observation model class. Currently
                    'Poisson' -> `PoissonObs` is implemented
        data_path: The local directory where the generated data should be stored. If None,
                    don't store shit.
        save_data_file: The name of the file to hold your data
        Nsamps: Number of trials to generate
        NTbins: Number of time steps to run.
        xDim: The dimensions of the latent space.
        yDim: The dimensions of the data.
        write_params_file: Would you like the parameters with which this data has been 
                    generated to be saved to a separate txt file?
    """    
    print('Generating some fake data...!\n')
    lat_mod_classes = {'llinear' : LocallyLinearEvolution}
    gen_mod_classes = {'Poisson' : PoissonObs, 'Gaussian' : GaussianObs}

    evolution_class = lat_mod_classes[lat_mod_class]
    generator_class = gen_mod_classes[gen_mod_class]

    if data_path:
        if not type(save_data_file) is str:
            raise ValueError("`save_data_file` must be string (representing the name of your file) "
                             "if you intend to save the data (`data_path` is not None)")
        if not os.path.exists(data_path): os.makedirs(data_path)
        if write_params_file:
            write_option_file(data_path)
    
    # Generate some fake data for training, validation and test
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            xDim = params.xDim
            yDim = params.yDim
            if not Nsamps: Nsamps = params.genNsamps
            if not NTbins: NTbins = params.genNTbins
            
            X = tf.placeholder(DTYPE, shape=[None, None, xDim], name='X')
            Y = tf.placeholder(DTYPE, shape=[None, None, yDim], name='Y')
            latm = evolution_class(X, params)
            genm = generator_class(Y, X, params, latm, is_out_positive=True)
        
            Nsamps_train = int(4*Nsamps/5)
            valid_test = int(Nsamps/10)
            sess.run(tf.global_variables_initializer())
            Ydata, Xdata = genm.sample_XY(sess, 'X:0', Nsamps=Nsamps, NTbins=NTbins,
                                          with_inflow=True)
            Ytrain, Xtrain = Ydata[:Nsamps_train], Xdata[:Nsamps_train]
            Yvalid, Xvalid = Ydata[Nsamps_train:-valid_test], Xdata[Nsamps_train:-valid_test]
            Ytest, Xtest = Ydata[valid_test:], Xdata[valid_test:]
            
            # If xDim == 2, draw a cool path plot
            if draw_quiver and xDim == 2:
                latm.plot_2Dquiver_paths(sess, Xdata, 'X:0', rlt_dir=data_path,
                                     with_inflow=True, savefig=savefigs)
            if draw_heat_maps:
                maxY = np.max(Ydata)
                for i in range(1):
                    plt.figure()
                    sns.heatmap(Ydata[i].T, yticklabels=False, vmax=maxY).get_figure()
                    if savefigs:
                        plt.savefig(data_path + "heat" + str(i) + ".png")
                    else:
                        plt.show()
                        plt.pause(0.001)
                        input('Press Enter to continue.')
                        plt.close()
            
    if data_path:
        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Xtrain' : Xtrain, 'Xvalid' : Xvalid,
                    'Ytest' : Ytest, 'Xtest' : Xtest}
        with open(data_path + save_data_file, 'wb+') as data_file:
            pickle.dump(datadict, data_file)
    
        if params.save_to_vind:
            with open(data_path + save_data_file + '_vind', 'wb+') as data_file:
                pickle.dump(datadict, data_file, protocol=2)
            
    return Ydata, Xdata


def main(_):
    """
    Launches this whole zingamajinga.
    """
    data_path = params.local_data_dir + params.this_data_dir
    rlt_dir = params.local_rlt_dir + params.this_data_dir + addDateTime() + '/'
    if params.mode == 'generate':
        generate_fake_data(lat_mod_class=params.lat_mod_class,
                           gen_mod_class=params.gen_mod_class,
                           params=params,
                           data_path=data_path,
                           save_data_file=params.save_data_file,
                           Nsamps=params.genNsamps,
                           NTbins=params.genNTbins,
                           write_params_file=True,
                           draw_quiver=True,
                           draw_heat_maps=True,
                           savefigs=True)
    if params.mode == 'train':
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(graph=graph)
            with sess.as_default():
                with open(data_path+params.save_data_file, 'rb+') as f:
                    # Set encoding='latin1' for python 2 pickled data
                    datadict = pickle.load(f, encoding='latin1') if params.is_py2 else pickle.load(f)
                    Ytrain = datadict['Ytrain']
                    Yvalid = datadict['Yvalid']

                params.yDim = Ytrain.shape[-1]
                opt = Optimizer_TS(params)
                
                sess.run(tf.global_variables_initializer())            
                opt.train(sess, rlt_dir, Ytrain, Yvalid)

    
if __name__ == '__main__':
    tf.app.run()



