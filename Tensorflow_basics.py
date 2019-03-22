import tensorflow as tf

## Defining Constants
a = tf.constant("a")
b = tf.constant(2)
c = tf.constant(3)
d = b + c
sess = tf.Session()
sess.run([a,b,c,d])
e = sess.run([b])
f = b
print("e = sess.run([b]): type is ",type(e))
print("f = b: type is ",type(f))
print("a: ",sess.run([a]),"b: ",sess.run([b]),"c: ",sess.run([c]),"d: ",sess.run([d]))

## Defining Variables
weights1 = tf.Variable(tf.random_normal([2,3],mean = 2, stddev = 0.5, seed = 1),name = 'weights1', trainable = True)
weights2 = tf.Variable(tf.zeros([2,3]),name = 'weights2', trainable = True)
weights3 = tf.Variable(tf.ones([2,3]),name = 'weights3', trainable = True)
weights4 = tf.Variable(tf.truncated_normal([2,3],mean = 2, stddev = 0.5, seed = 1),name = 'weights4', trainable = True)
weights5 = tf.Variable(tf.random_uniform([2,3],minval = 0.5, maxval = 2),name = 'weights5', trainable = True)
init = tf.global_variables_initializer()
sess.run([init])
sess.run([weights1,weights2,weights3,weights4,weights5])

## Defining Placeholders
x = tf.placeholder(tf.float32, name = 'x', shape = [20,None])
y = tf.placeholder(tf.float32, name = 'y', shape = [None,20])
z = tf.matmul(x,y)
print("x: " ,x)
print("y: ", y)
print("z: ", z)

