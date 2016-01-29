'''An implementation of ensemble selection.'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# TODO: implement another early stopping condition (early stopping via
# validation?)

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

def printCSV(predictions):
	'''Prints (to standard output) a CSV in the Kaggle format.'''
	print "Id,Prediction"
	for i in range(len(predictions)):
		print "%i,%i" % (i + 1, predictions[i])

def split(training, labels):
	'''Splits the training set training randomly into a training set and a
	validation set for use in ensemble selection.'''
	# First couple the training and labels so we can keep them together.
	for i in range(len(training)):
		training[i].append(labels[i]);
	length = len(training)
	half = np.floor(length / 2)
	permuted = np.random.permutation(training)
	train = permuted[:half]
	validate = permuted[half:]
	# Decouple the training and the labels.
	train = np.ndarray.tolist(train)
	# print len(train)
	validate = np.ndarray.tolist(validate)
	# print len(validate)
	tlabels = []
	vlabels = []
	for i in range(len(train)):
		tlabels.append(train[i].pop())
	for i in range(len(validate)):
		vlabels.append(validate[i].pop())
	# print len(tlabels)
	# print len(vlabels)
	return (train, tlabels, validate, vlabels)

def ensembleSelection(models, validation, labels, iterations = 10):
	'''Given a list of models and validation data, does ensemble election on
	the given models. Returns a list of the weights of each model in the
	ensemble. The default behavior of this function is to run the ensemble
	selection for a set number of iterations (set at 100).'''
	# We initially have no models.
	length = len(models)
	weights = [0] * length
	for i in range(iterations): # Replace with some early stopping code.
		max_score = -1
		max_index = -1
		for i in range(length):
			# Assumes the model is from sklearn
			score = models[i].score(validation, labels)
			if score > max_score:
				max_score = score
				max_index = i
		# Add a copy of the model to our ensemble.
		weights[i] += 1
	return weights


def predictEnsemble(models, weights, test):
	'''Scores the ensemble described by models and weights on the validation
	set given by training and labels. Assumes we are doing binary
	classification.'''
	length = len(test)
	pclass0 = np.zeros(length)
	pclass1 = np.zeros(length)
	for i in range(len(models)):
		# First element is 0, second element is 1.
		probs = models[i].predict_proba(test)
		p1 = probs[:, 0]
		p2 = probs[:, 1]
		pclass0 = np.add(pclass0, p1)
		pclass1 = np.add(pclass1, p2)
	# Output the results. I'm just seeing which summed probability is greater,
	# which might be wrong RIP.
	predictions = []
	for i in range(length):
		if pclass0[i] > pclass1[i]:
			predictions.append(0)
		else:
			predictions.append(1)
	return predictions

if __name__ == '__main__':
	(training, labels) = parseTraining('training_data.txt', '|')
	test = parseTest('testing_data.txt', '|')
	# clf = RandomForestClassifier()
	# clf = clf.fit(training, labels)
	# predict = clf.predict(test)
	# prob = clf.predict_proba(test)
	# print predict[0]
	# print prob[0]

	# Some sample code. This module is mainly meant to be used in other code.
	# Split the data into a training and validation set.
	(train, tlabels, validate, vlabels) = split(training, labels)
	models = []
	# Train a bunch of classifiers with varying max depth.
	# print len(train)
	# print len(tlabels)
	# print len(validate)
	# print len(vlabels)
	for i in range(1, 11):
		clf = RandomForestClassifier(max_depth = i)
		clf.fit(train, tlabels)
		models.append(clf)
	# Now perform ensemble selection.
	weights = ensembleSelection(models, validate, vlabels)
	# Get prediction from our model.
	predictions = predictEnsemble(models, weights, test)
	printCSV(predictions)