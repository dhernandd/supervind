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
    Compute the Cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block 
        off-diagonal matrix

    Outputs: 
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky
    """
    def compute_chol(LC, AB_2xdxd):
        L_dxd = LC[0]
        A_dxd, B_dxd = AB_2xdxd[0], AB_2xdxd[1]
        C_dxd = tf.matmul(B_dxd, tf.matrix_inverse(L_dxd), 
                      transpose_a=True, transpose_b=True)
        D = A_dxd - tf.matmul(C_dxd, C_dxd, transpose_b=True)
        L_dxd = tf.cholesky(D)
        return [L_dxd, C_dxd]
        
    L1_dxd = tf.cholesky(A_Txdxd[0])
    C1_dxd = tf.zeros_like(B_Tm1xdxd[0], dtype=tf.float64)
    
    result_2xTm1xdxd = tf.scan(fn=compute_chol, elems=[A_Txdxd[1:], B_Tm1xdxd],
                               initializer=[L1_dxd, C1_dxd])

    AChol_Txdxd = tf.concat([tf.expand_dims(L1_dxd, 0), result_2xTm1xdxd[0]], 
                            axis=0)    
    BChol_Tm1xdxd = result_2xTm1xdxd[1]
    
    return AChol_Txdxd, BChol_Tm1xdxd


def blk_chol_inv(A_Txdxd, B_Tm1xdxd, b_Txd, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a block-bi-
    diagonal triangular matrix - only the first lower/upper off-diagonal block
    is nonvanishing.
    
    This function will be used to solve the equation Mx = b where M is a
    block-tridiagonal matrix due to the fact that M = C^T*C where C is block-
    bidiagonal triangular.
    
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower) 
        1st block off-diagonal matrix
     
    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the 
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve 
          the problem C^T x = b with a representation of C.) 
 
    Outputs: 
    x - solution of Cx = b
    """
    # Define a matrix-vector dot product because the tensorflow developers feel
    # this is beneath them.
    tf_dot = lambda M, v : tf.reduce_sum(tf.multiply(M, v), axis=1)
    if transpose:
        A_Txdxd = tf.transpose(A_Txdxd, [0,2,1])
        B_Tm1xdxd = tf.transpose(B_Tm1xdxd, [0,2,1])
    
    # Whether B is lower or upper doesn't matter. The function to be passed to
    # scan is the same.
    def step(x_d, ABb_2x_):
        A_dxd, B_dxd, b_d = ABb_2x_[0], ABb_2x_[1], ABb_2x_[2]
        return tf_dot(tf.matrix_inverse(A_dxd),
                         b_d - tf_dot(B_dxd, x_d))
    if lower:
        x0_d = tf_dot(tf.matrix_inverse(A_Txdxd[0]), b_Txd[0])
        result_Tm1xd = tf.scan(fn=step, elems=[A_Txdxd[1:], B_Tm1xdxd, b_Txd[1:]], 
                             initializer=x0_d)
        result_Txd = tf.concat([tf.expand_dims(x0_d, axis=0), result_Tm1xd], axis=0)
    else:
        xN_d = tf_dot(tf.matrix_inverse(A_Txdxd[-1]), b_Txd[-1])
        result_Tm1xd = tf.scan(fn=step, 
                             elems=[A_Txdxd[:-1][::-1], B_Tm1xdxd[::-1], b_Txd[:-1][::-1]],
                             initializer=xN_d )
        result_Txd = tf.concat([tf.expand_dims(xN_d, axis=0), result_Tm1xd],
                               axis=0)[::-1]

    return result_Txd 


if __name__ == '__main__':
    # Test blk_tridiag_chol
    
    # Note that the matrices forming the As here are symmetric. Also, I have
    # chosen the entries wisely - cuz I'm a wise guy - so that the overall
    # matrix is positive definite as required by the algo.
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

    As = tf.stack([tF, tC, tE, tG])
    Bs = tf.stack([tf.transpose(tB), tf.transpose(tD), tf.transpose(tB)])
    
    AChol, BChol = blk_tridiag_chol(As, Bs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#         sess.run([AChol, BChol])
        
    # Test blk_chol_inv
    
    # The matrices forming the As are in lower triangular form. 
    npA = np.mat('1  .9; .9 4')
    npB = .01*np.mat('2  7; 7 4')
    npC = np.mat('3.0  0.0; 0.0 1.0')
    npD = .01*np.mat('7  2; 9 3')
    npE = .01*np.mat('2  0; 4 3')
    npF = .01*np.mat('1  0; 2 7')
    npG = .01*np.mat('3  0; 8 1')
    npZ = np.mat('0 0; 0 0')
    lowermat = np.bmat([[npF,     npZ, npZ,   npZ],
                           [npB.T,   npC, npZ,   npZ],
                           [npZ,   npD.T, npE,   npZ],
                           [npZ,     npZ, npB.T, npG]])
    rA = tf.get_variable('rA', initializer=npA)
    rB = tf.get_variable('rB',initializer=npB)
    rC = tf.get_variable('rC',initializer=npC)
    rD = tf.get_variable('rD',initializer=npD)
    rE = tf.get_variable('rE',initializer=npE)
    rF = tf.get_variable('rF',initializer=npF)
    rG = tf.get_variable('rG',initializer=npG)
    npb = np.mat('1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0')
    rb = tf.get_variable('b', initializer=npb)

    theD = tf.stack([rF, rC, rE, rG])
    theOD = tf.stack([tf.transpose(rB), tf.transpose(rD), tf.transpose(rB)])
    
    ib = blk_chol_inv(theD, theOD, rb)
    res = blk_chol_inv(theD, theOD, ib, lower=False, transpose=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
        print(res)
