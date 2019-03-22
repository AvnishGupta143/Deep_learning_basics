
# coding: utf-8

# In[50]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[51]:

#Create Placeholders
x = tf.placeholder(tf.float32, [1,None] , name = 'x')
y = tf.placeholder(tf.float32, [1,None] , name = 'y')

#Initialise Parameters
W = tf.Variable([2],dtype = tf.float32, name = 'W')
b = tf.Variable([1],dtype = tf.float32, name = 'b')
num_iterations = 1000
X_train = np.array([[10,20,30,40]])
Y_train = np.array([[20,40,60,80]])
costs = []
learning_rate = 0.0001


# In[52]:

#Forward Propagation
Z = tf.add(tf.multiply(W,x),b)
print(Z.shape)

#Compute Cost
cost = tf.reduce_mean(tf.square(Z-y))
#cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Z,labels = y)
 
#Backward Propagation
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)


# In[53]:

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(num_iterations):
        _, epoch_cost = sess.run([optimizer,cost],feed_dict = {x:X_train,y:Y_train})
        costs.append(epoch_cost)
    
    parameters = sess.run([W,b])
        
#plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('itrations')
plt.title("Learning Rate" + str(learning_rate))

print(parameters)


# In[ ]:



