'''Kaggle Project First Try: Parsing the input'''

import math
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier

# Holds the words corresponding to each coordinate in the bag of words
# representation.
names = []

# Holds the number of shallow decision trees which we will boost with.
nboost = 50
# Holds the number of deep decision trees we will bag.
nbagged = 50
# For random stuff like extra random trees.
nrand = 50

def parseTraining(filename, delim):
	'''Parses the file at filename with delimiter str and returns
	the a tuple whose first element contains the feature matrix and whose
	second element contains the label vector. Assumes the label is at the
	end of each line.'''
	f = open('%s' % filename, 'r');
	training = []
	labels = []
	length = 0;
	for line in f:
		 g = line.strip().split(delim)
		 if g[0] == 'thi':
		 	# names = g; # Holds the names in a global variable just for kicks.
		 	length = len(g)
		 else:
		 	g = [int(i) for i in g] # Turn the strings into ints.
		 	training.append(g[:length - 1])
		 	# Note that the labels are 0 for against and 1 for support.
		 	labels.append(g[length - 1])
	f.close()
	return (training, labels)

def parseTest(filename, delim):
	'''Parses the file at filename with delimiter delim. For some reason the
	training set doesn't have labels.'''
	f = open('%s' % filename, 'r');
	training = []
	length = 0;
	for line in f:
		 g = line.strip().split(delim)
		 if g[0] != 'thi':
		 	g = [int(i) for i in g] # Turn the strings into ints.
		 	training.append(g)
	return training

def predict(classifier, test):
	'''Given a classifier and a test data set, outputs the data in the way
	Kagglorg expects, which is a 1-indexed ID followed by the label.'''
	predictions = classifier.predict(test)
	print "Id,Prediction"
	for i in range(0, len(predictions)):
		print "%i,%i" % (i + 1, predictions[i])

def multWeights(experts, labels, eta):
	'''Implements the multiplicative weights algorithm. Returns a weight for
	each expert. Is regularized with the entropy penalty function.'''
	# Initialize the weights to be uniform.
	weights = [1] * len(experts)
	for i in range(len(experts[0])):
		for j in range(len(experts)):
			score = 0;
			prediction = 0;
			score += experts[j][i] * weights[j]
			if score > 0:
				prediction = 1;
			weights[j] *= math.pow(math.e, -1 * eta * label)
	return weights



if __name__ == '__main__':
	'''This file is meant to be run with something like
	   python parse.py > predict.txt
	   since it just prints the prediction to standard output otherwise.'''
	(training, labels) = parseTraining('training_data.txt', '|')
	test = parseTest('testing_data.txt', '|')
	# Boosted Decision Trees
	# We train a number of trees of max depth 3 (the default argument). Use
	# 'exponential' to tell the class to use the AdaBoost algoritum.
	boost1 = GradientBoostingClassifier(loss = 'exponential', n_estimators = nboost)
	boost1 = boost1.fit(training, labels)

	# AdaBoost blaster
	# boost2 = AdaBoostClassifier(n_estimators = nboost)
	# boost2 = boost2.fit(training, labels)

	# Trains some EXTRA random decision trees.
	rand = ExtraTreesClassifier(n_estimators = nrand)
	rand = rand.fit(training, labels)

	# Trains a number of bagged deep decision trees with default arguments to
	# RandomForestClassifier except n_estimates = nbagged
	bag = RandomForestClassifier(n_estimators = nbagged)
	bag = bag.fit(training, labels)

	# Aggregate the classifiers with soft voting.
	eclf = VotingClassifier(estimators = [('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'hard')
	eclf.fit(training, labels)
	predict(eclf, test)
