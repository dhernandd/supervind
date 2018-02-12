import tensorflow as tf
import numpy as np


def gen_random():
    for _ in range(1):
        print(np.random.randn())
 
class Test1():
    def test_simple(self):
        with tf.Session():
            gen_random()
             
    def test_simple1(self):
        with tf.Session():
            gen_random()
     
     
class Test2(tf.test.TestCase):
    def test_simple(self):
        with tf.Session():
            gen_random()
             
    def test_simple1(self):
        with tf.Session():
            gen_random()
             
# a = np.array([1.0,2.0,3.0])
# b = np.array([1.0,2.0,3.0])
# 
# A = tf.get_variable('A', initializer=a)
# B = tf.get_variable('B', initializer=b)
# 
# with tf.Graph().as_default() as g1:
#     with tf.variable_scope('foo', reuse=tf.AUTO_REUSE):
#         x = tf.placeholder(dtype=tf.float64, shape=[2], name='x')
#         c = tf.get_variable('c', initializer=tf.cast(1.0, tf.float64))
#         y = tf.identity(2*x, 'y')
#         
#         z = tf.identity(3*x*c, 'z')
#         g1_def = g1.as_graph_def()
#         z1, = tf.import_graph_def(g1_def, input_map={'foo/x:0' : y}, return_elements=["foo/z:0"],
#                                   name='z1')
#         init_op = tf.global_variables_initializer()
#         print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='foo'))
# 
# 
# with tf.Session(graph=g1) as sess:
#     sess.run(init_op)
#     print(sess.run(z, feed_dict={'foo/x:0' : np.array([1.0, 2.0])}))
#     print(sess.run(tf.report_uninitialized_variables()))
#     z1 = sess.run(z1, feed_dict={'foo/x:0' : np.array([1.0, 2.0])})
    
#     y = sess.run(y, feed_dict={'x:0' : np.array([1.0, 2.0])})
#     print('y:', y)
    
#     print(z1)


# print(A, B)
# print(tf.constant(0.0))
# aux_fn = lambda _, seqs : seqs[0] + seqs[1]
# C = tf.scan(fn=aux_fn, elems=[A, B], initializer=tf.cast(tf.constant(0.0), tf.float64))
# 
# elems = np.array([1, 2, 3, 4, 5, 6])
# initializer = np.array(0)
# sum_one = tf.scan(lambda _, x: x[0] - x[1], (elems + 1, elems), initializer)
# 
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(C))
# print(sess.run(sum_one))



if __name__ == '__main__':
    t = Test1()
    t.test_simple()
    t.test_simple()
    tf.test.main()
    
