'''Kaggle Project First Try: Parsing the input'''

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn import tree, cross_validation, svm, feature_selection, naive_bayes, pipeline, preprocessing, grid_search, cluster, decomposition

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

# 	'''This file is meant to be run with something like
# 	   python parse.py > predict.txt
# 	   since it just prints the prediction to standard output otherwise.'''
# 	(training, labels) = parseTraining('training_data.txt', '|')
# 	test = parseTest('testing_data.txt', '|')
# 	xTrain, xValidate, yTrain, yValidate = cross_validation.train_test_split(training, labels, test_size=0.1)
# 	# Boosted Decision Trees
# 	# We train a number of trees of max depth 3 (the default argument). Use
# 	# 'exponential' to tell the class to use the AdaBoost algoritum.
# 	boost1 = GradientBoostingClassifier(loss = 'exponential', n_estimators = nboost)
# 	boost1 = boost1.fit(training, labels)

# 	# AdaBoost blaster
# 	# boost2 = AdaBoostClassifier(n_estimators = nboost)
# 	# boost2 = boost2.fit(training, labels)

# 	# Trains some EXTRA random decision trees.
# 	rand = ExtraTreesClassifier(n_estimators = nrand)
# 	rand = rand.fit(training, labels)

# 	# Trains a number of bagged deep decision trees with default arguments to
# 	# RandomForestClassifier except n_estimates = nbagged
# 	bag = RandomForestClassifier(n_estimators = nbagged)
# 	bag = bag.fit(training, labels)

# 	# Trains some decision tree classifiers and picks the best one
# 	train_err = []
# 	val_err = []
# 	ind = 0
# 	max_score = 0
# 	for i in np.arange(1, 200, 5):
# 	# Specify tree params
# 		dTree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = i)
# 		# Calculate Error
# 		scores = cross_validation.cross_val_score(dTree, training, labels, cv = 5)
# 		mean = scores.mean()
#         if mean > max_score:
# 	    	max_score = mean
# 	    	ind = i
# 	    	print("Tree Accuracy: %0.2f (+/- %0.2f)" % (mean, scores.std() * 2))

# 	bestTree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = ind)
# 	bestTree.fit(training, labels)

# 	#SVM
# 	ind = 0
# 	max_score = 0
# 	for i in np.arange(0.3, 2, 0.1):
# 	    supp_lin = svm.LinearSVC(C=i, penalty='l1', dual=False)
# 	    scores = cross_validation.cross_val_score(supp_lin, training, labels, cv=5)
# 	    mean = scores.mean()
# 	    if mean > max_score:
# 	    	max_score = mean
# 	    	ind = i
# 	    	print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 	supp_lin = svm.LinearSVC(C=ind, penalty='l1', dual=False)
# 	supp_lin.fit(training, labels)

# 	# more Decision trees: Mod max depth param
# 	train_err = []
# 	val_err = []
# 	ind = 0
# 	max_score = 0
# 	for i in np.arange(2, 202, 5):
# 	# Specify tree params
# 		dTree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = i)
# 		# Calculate Error
# 		scores = cross_validation.cross_val_score(dTree, training, labels, cv = 5)
# 		mean = scores.mean()
#         if mean > max_score:
# 	    	max_score = mean
# 	    	ind = i
# 	    	print("TreeD Accuracy: %0.2f (+/- %0.2f)" % (mean, scores.std() * 2))

# 	bestTreeD = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = ind)
# 	bestTreeD.fit(training, labels)

# 	ind = 0
# 	max_score = 0
# 	# Nearest Neighbors
# 	for i in range(2, 20):
# 		neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
# 		scores = cross_validation.cross_val_score(neigh, training, labels, cv=5)
# 		mean = scores.mean()
# 		if mean > max_score:
# 			max_score = mean
# 			ind = i
# 			print("NearestNeighbors Accuracy: %0.2f (+/- %0.2f)" % (mean, scores.std() * 2))

# 	bestNeigh = neighbors.KNeighborsClassifier(n_neighbors=ind)
# 	bestNeigh.fit(training, labels)

# 	# Aggregate the classifiers with soft voting.
# 	eclf = VotingClassifier(estimators = [('NearestNeighbors', bestNeigh), ('DecisionTreeD', bestTreeD), ('SVM', supp_lin), ('DecisionTree', bestTree), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'hard')
# 	end_scores = cross_validation.cross_val_score(eclf, training, labels, cv=5)
# 	print "The validation error is %f" % (1 - end_scores.mean())
# 	print "The std is %f" % end_scores.std()
# 	eclf.fit(training, labels)
# 	predict(eclf, test)
# >>>>>>> d73c6388a3bce1b9366ccd35522f3d3dc5baf229

#     def transform(X, y=None):
#         X = np.add(np.amin(X), X)
#         return X
