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

import numpy as np

import tensorflow as tf

def variable_in_cpu(name, shape, initializer):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=tf.float64, 
                              initializer=initializer)
    return var


def blk_tridiag_chol(A_Txdxd, B_Tm1xdxd):
    """
    """
    A_Tx1xdxd, B_Tm1x1xdxd = tf.expand_dims(A_Txdxd, axis=1), tf.expand_dims(B_Tm1xdxd, axis=1)
    AB_Tm1x2xdxd = tf.concat([A_Tx1xdxd[1:], B_Tm1x1xdxd], axis=1)
    def compute_chol(LC, AB_2xdxd):
        L = LC[0]
        A_dxd, B_dxd = AB_2xdxd[0], AB_2xdxd[1]
        C = tf.matmul(B_dxd, tf.matrix_inverse(L), transpose_a=True, transpose_b=True)
        D = A_dxd - tf.matmul(C, C, transpose_b=True)
        L = tf.cholesky(D)
        LC = tf.concat([tf.expand_dims(L,0), tf.expand_dims(C,0)], 0)
        return LC
    
    L1_1xdxd = tf.expand_dims(tf.cholesky(A_Txdxd[0]), axis=0)
    C1_1xdxd = tf.expand_dims(tf.zeros_like(B_Tm1xdxd[0], dtype=tf.float64), axis=0)
    LC_2xdxd = tf.concat([L1_1xdxd, C1_1xdxd], axis=0)
    
    result_Tm1x2xdxd = tf.scan(fn=compute_chol, elems=AB_Tm1x2xdxd, 
                               initializer=LC_2xdxd)
    AChol_Txdxd = tf.concat([L1_1xdxd, result_Tm1x2xdxd[:,0,:,:]], axis=0)
    BChol_Tm1xdxd = result_Tm1x2xdxd[:,1,:,:]
    
    return AChol_Txdxd, BChol_Tm1xdxd


if __name__ == '__main__':
    npA = np.mat('1  .9; .9 4')
    npB = .01*np.mat('2  7; 7 4')
    npC = np.mat('3.0  0.0; 0.0 1.0')
    npD = .01*np.mat('7  2; 9 3')
    npE = np.mat('2  0.4; 0.4 3')
    npF = np.mat('1  0.2; 0.2 7')
    npG = np.mat('3  0.8; 0.8 1')

    npZ = np.mat('0 0; 0 0')

    lowermat = np.bmat([[npF,     npZ, npZ,   npZ],
                           [npB.T,   npC, npZ,   npZ],
                           [npZ,   npD.T, npE,   npZ],
                           [npZ,     npZ, npB.T, npG]])


    tA = tf.get_variable('tA', initializer=npA)
    tB = tf.get_variable('tB',initializer=npB)
    tC = tf.get_variable('tC',initializer=npC)
    tD = tf.get_variable('tD',initializer=npD)
    tE = tf.get_variable('tE',initializer=npE)
    tF = tf.get_variable('tF',initializer=npF)
    tG = tf.get_variable('tG',initializer=npG)

    theD = tf.stack([tF, tC, tE, tG])
    theOD = tf.stack([tf.transpose(tB), tf.transpose(tD), tf.transpose(tB)])
    
    A, B = blk_tridiag_chol(theD, theOD)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(A))
        print(sess.run(B))
