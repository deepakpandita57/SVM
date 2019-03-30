#!/usr/bin/python3

# Author: Deepak Pandita
# Date created: 02 Feb 2018

from pandita_deepak_hw3 import train_svm, getAccuracy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def main():
	#file paths
	train_file = '/data/adult/a7a.train'
	dev_file = '/data/adult/a7a.dev'
	test_file = '/data/adult/a7a.test'

	#learning rate
	eta = 0.1
	#no. of epochs and capacity
	epochs  = 5

	print("EPOCHS: "+str(epochs))
	print("LEARNING_RATE: "+str(eta))

	#Read train file
	#print('Reading file: '+train_file)
	f = open(train_file)
	train_examples = f.readlines()
	f.close()

	#Read test file
	#print('Reading file: '+test_file)
	t = open(test_file)
	test_examples = t.readlines()
	t.close()

	#Read dev file
	#print('Reading file: '+dev_file)
	d = open(dev_file)
	dev_examples = d.readlines()
	d.close()

	#Call SVM
	cs = []
	initial_c = 0.001
	while initial_c<=10000:
		cs.append(initial_c)
		initial_c = initial_c*1.5
	print("No. of Cs: "+str(len(cs)))

	test_accuracies = []
	dev_accuracies = []
	for c in cs:
		#weights (There are 123 features in the data)
		w = np.zeros(123)
		#bias term
		b = 0
		learned_weights, learned_bias = train_svm(train_examples, epochs, c, eta, w, b)

		test_accuracy = getAccuracy(test_examples, learned_weights, learned_bias)
		test_accuracies.append(test_accuracy)
		dev_accuracy = getAccuracy(dev_examples, learned_weights, learned_bias)
		dev_accuracies.append(dev_accuracy)

	#print("Cs: "+str(cs))
	#print("TEST_ACCURACIES: "+str(test_accuracies))
	#print("DEV_ACCURACIES: "+str(dev_accuracies))
	plt.plot(cs,test_accuracies,label='Test Accuracy')
	plt.plot(cs,dev_accuracies,label='Dev Accuracy')

	plt.xscale('log')
	plt.xlabel("C")
	plt.ylabel("Accuracy")
	plt.title("Plot of accuracy on test and dev set with C")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()