import tensorflow as tf
import numpy as np


var1 = tf.constant([[1,2]])
var2 = tf.constant([[2],[3]])

result = tf.matmul(var1,var2)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)
 