import tensorflow as tf 

def layer(Input, weight_shape, bias_shape):
	weight_init = tf.random_uniform_initializer(minval = -1, maxval = 1)
	bias_init = tf.constant_initializer(value = 0)

	#tf.get_variable(<name>, <shape>, <initializer>)
	W = tf.get_variable("W", weight_shape, initializer = weight_init)
	b = tf.get_variable("b", bias_shape, initializer = bias_init)

	return tf.matmul(Input,W) + b

def network(Input):
	#tf.variable_scope(<scope_name>)
	with tf.variable_scope("layer_1"):
		output_1 = layer(Input, [784,100], [100])

	with tf.variable_scope("layer_2"):
		output_2 = layer(output_1, [100,50], [50])

	with tf.variable_scope("layer_3"):
		output_3 = layer(output_2, [50,10], [10])	

	return output_3

def call_network():
	with tf.variable_scope("shared_vatiables") as scope:
		i_1 = tf.placeholder(tf.float32, [1000,784], name = "i_1")
		network(i_1)
		scope.reuse_variables()
		i_2	= tf.placeholder(tf.float32, [1000,784], name = "i_2")
		network(i_2)

def main():
	call_network()

if __name__ == '__main__':
	main()