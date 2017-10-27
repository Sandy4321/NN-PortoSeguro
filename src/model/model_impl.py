# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:50:19 2017

@author: andersonduran
"""
import theano
import theano.tensor as T
import numpy as np

def define_predict(W1, b1, W2, b2, W3, b3):
	#--------------------------------------------------------------------
	#
	#Forward propagation
	#
	#--------------------------------------------------------------------
	A0 = T.dmatrix('A0')
	
	Z1 = T.dot(W1, A0) + b1
	A1 = T.nnet.relu(Z1)
	
	Z2 = T.dot(W2, A1) + b2
	A2 = T.nnet.relu(Z2)
	
	Z3 = T.dot(W3, A2) + b3
	A3 = T.nnet.sigmoid(Z3)

	#--------------------------------------------------------------------
	#
	#Actual computation
	#
	#--------------------------------------------------------------------
	return theano.function([A0], A3)

def define_model(layers):
	#--------------------------------------------------------------------
	#
	# Parameters initialization
	#
	#--------------------------------------------------------------------
	W1 = theano.shared(np.random.randn(layers[1], layers[0])*0.1)
	W2 = theano.shared(np.random.randn(layers[2], layers[1])*0.1)
	W3 = theano.shared(np.random.randn(layers[3], layers[2])*0.1)

	b1 = theano.shared(0.)
	b2 = theano.shared(0.)
	b3 = theano.shared(0.)

	#--------------------------------------------------------------------
	#
	#Forward propagation
	#
	#--------------------------------------------------------------------
	A0 = T.dmatrix('A0')
	
	Z1 = T.dot(W1, A0) + b1
	A1 = T.nnet.relu(Z1)
	
	Z2 = T.dot(W2, A1) + b2
	A2 = T.nnet.relu(Z2)
	
	Z3 = T.dot(W3, A2) + b3
	A3 = T.nnet.sigmoid(Z3)

	#--------------------------------------------------------------------
	#
	#Cost function - cross entropy
	#
	#--------------------------------------------------------------------
	labels = T.dmatrix('labels')
	cost = -1/A0.shape[1]*(labels*T.log(A3) + (1-labels)*T.log(1-A3)).sum()

	#--------------------------------------------------------------------
	#
	#Gradient descent
	#
	#--------------------------------------------------------------------
	dW1, dW2, dW3, db1, db2, db3 = T.grad(cost, [W1, W2, W3, b1, b2, b3])
	
	alpha = T.dscalar('alpha')

	#--------------------------------------------------------------------
	#
	#Update parameters
	#
	#--------------------------------------------------------------------
	return theano.function(
		inputs=[A0, labels, alpha],
		outputs=[cost, W1, W2, W3, b1, b2, b3],
		updates=[
			[W1, W1 - alpha*dW1],
			[W2, W2 - alpha*dW2],
			[W3, W3 - alpha*dW3],
			[b1, b1 - alpha*db1],
			[b2, b2 - alpha*db2],
			[b3, b3 - alpha*db3]
		])