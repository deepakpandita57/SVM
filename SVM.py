#!/usr/bin/python3

# Author: Deepak Pandita
# Date created: 02 Feb 2018

import numpy as np
import argparse

#This function trains a Support Vector Machine on training examples with given epochs, capacity, learning rate, initial weights and bias
#The function returns the learned weights and bias
def train_svm(examples, epochs, c, eta, w, b):
	#print('Running SVM...')
	N = len(examples)
	epoch = 0
	while(epoch < epochs):
		epoch += 1
		#print('Epoch: ' + str(epoch))
		
		for line in examples:
			tokens = line.strip().split(' ')
			y = float(tokens[0])	#label
			instance = tokens[1:]
			x = np.zeros(123)
			for token in instance:
				feature = int(token.split(":")[0])
				value = float(token.split(":")[1])
				#print feature
				x[feature-1] = value
			if (1 - y * (sum(w * x) + b)) >= 0:
				w = w - eta * ((w / N) - c * y * x)
				b = b + eta * y * c
			else:
				w = w - eta * (w / N)
	return w,b

#This function predicts the label on given examples using given weights, bias and returns the accuracy
def getAccuracy(examples, w, b):
	correct = 0
	for line in examples:
		tokens = line.strip().split(' ')
		y = float(tokens[0])	#label
		instance = tokens[1:]
		x=np.zeros(123)
		for token in instance:
			feature = int(token.split(":")[0])
			value = float(token.split(":")[1])
			#print feature
			x[feature-1] = value
		if y*(sum(w * x) + b) > 0:
			correct += 1
	accuracy = float(correct)/len(examples)
	return accuracy

def main():
	#using optional parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', action="store", help = "No. of epochs to run", type = int)
	parser.add_argument('--capacity', help = "Capacity", action = "store", type = float)
	args = parser.parse_args()

	#file paths
	train_file = '/data/adult/a7a.train'
	dev_file = '/data/adult/a7a.dev'
	test_file = '/data/adult/a7a.test'


	#learning rate
	eta = 0.1

	#default no. of epochs and capacity
	epochs  = 1
	c = 0.868

	if args.epochs:
		epochs = args.epochs
	if args.capacity:
		c = args.capacity

	#weights (There are 123 features in the data)
	w = np.zeros(123)
	#bias term
	b = 0

	print("EPOCHS: "+str(epochs))
	print("CAPACITY: "+str(c))

	#Read train file
	#print('Reading file: '+train_file)
	f = open(train_file)
	train_examples = f.readlines()
	f.close()

	#Call SVM
	learned_weights, learned_bias = train_svm(train_examples, epochs, c, eta, w, b)

	#Accuracy on training set
	training_accuracy = getAccuracy(train_examples, learned_weights, learned_bias)
	print("TRAINING_ACCURACY: " + str(training_accuracy))

	#Read test file
	#print('Reading file: '+test_file)
	t = open(test_file)
	test_examples = t.readlines()
	t.close()

	#Accuracy on test set
	test_accuracy = getAccuracy(test_examples, learned_weights, learned_bias)
	print("TEST_ACCURACY: " + str(test_accuracy))

	#Read dev file
	#print('Reading file: '+dev_file)
	d = open(dev_file)
	dev_examples = d.readlines()
	d.close()

	#Accuracy on dev set
	dev_accuracy = getAccuracy(dev_examples, learned_weights, learned_bias)
	print("DEV_ACCURACY: " + str(dev_accuracy))

	#Print final bias and weights
	final_svm = []
	final_svm.append(learned_bias)
	for wt in learned_weights:
		final_svm.append(wt)
	print("FINAL_SVM: "+str(final_svm))

if __name__ == '__main__':
	main()