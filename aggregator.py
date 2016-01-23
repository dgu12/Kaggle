'''Implements several different meta-learning (aggregation) algorithms'''

import sys
import random

# Eta for the multiplicative weights algorithm.
eta = 0.5

def parseCSV(filename):
	'''Parses a CSV in the Kaggle format: The first line contains the comma-
	-separated string "Id,Prediction" and each subsequent lines contains 2
	comma-separated-values: the first contains the ID and the second contains
	the classification by the model. Returns the vector of predictions.'''
	f = open('%s' % filename, 'r')
	predictions = []
	for line in f:
		g = line.strip().split(',')
		if g[0] == "Id":
			continue
		g = [int(i) for i in g]
		predictions.append(g[1])
	return predictions

def printCSV(predictions):
	'''Prints (to standard output) a CSV in the Kaggle format.'''
	print "Id,Prediction"
	for i in range(len(predictions)):
		print "%i,%i" % (i + 1, predictions[i])

def majorityVote(models):
	'''Aggregates the data by taking a simple majority vote of the models. if
	there is a tie, it is broken randomly.'''
	random.seed()
	prediction = []
	for i in range(len(models[0])):
		yes = 0
		no = 0
		for j in range(len(models)):
			if models[j][i] == 1:
				yes += 1
			else:
				no += 1
		if yes > no:
			prediction.append(1)
		elif yes < no:
			prediction.append(0)
		else:
			prediction.append(random.randint(0, 1))
	return prediction

def multWeights(experts, eta, T):
	'''Implements the multiplicative weights algorithm. Returns a weight for
	each expert. Is regularized with the entropy penalty function.'''
	# Initialize the weights to be uniform.
	weights = [1] * len(experts)
	pass



if __name__ == '__main__':
	'''This module takes in command line arguments of the format
	python [prog] [-m/-w] [filename] [filename]...
	Assumes all of the command line arguments which are not flags are valid
	filenames fitting the Kaggle CSV format.'''
	models = []
	mflag = False;
	wflag = False;
	for i in range(1, len(sys.argv)):
		if sys.argv[i] == '-m':
			mflag = True
		else:
			models.append(parseCSV(sys.argv[i]))
	if True:
		printCSV(majorityVote(models))
