import tensorflow as tf

c = []	
for d in ['/gpu:0', '/gpu:1']: 
	with tf.device(d):
		a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
		b = tf.constant([1.0, 2.0], shape=[2, 1], name='b') 
		c.append(tf.matmul(a, b))
with tf.device('/cpu:0'): 
	sum = tf.add_n(c)
#If we would like TensorFlow to find another available device if the chosen device does not exist, we can pass the allow_soft_placement flag
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(sum)
