# NEAREST NEIGHBOUR CLASSIFIER
import numpy as np

class NearestNeighbour:
	
	def __init__(self):
		pass	

	# X is NxD where N is nu. of images and D is dimension.
	# Y is N x 1 array with labels
	# The function train tries to remember all the data
	def train(self,X,Y):
		self.Xtr = X
		self.Ytr = Y

	# Here X is N x D where N is no. of images to classify
	def predict(self,X):
		Ypred = np.zeros([X.shape[0],1],dtype = self.Ytr.dtype)
		for i in range(X.shape[0]):
			L1_norm = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
			min_index = np.argmin(L1_norm)
			Ypred[i] = self.Ytr[min_index]
		return Ypred

# Train  --> O(1)
# Test   --> O(N) 
# Here K=1, That is we are checking only for one nearest neighbour in the data set and assigning the label same as that nearest neighbour
# We can optimize by increasing the K, that is we will check for K nearest neighbours and assign label on the basis of majority in the labels of K nearest neighbours
