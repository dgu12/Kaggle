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
<<<<<<< HEAD
    '''This file is meant to be run with something like
       python parse.py > predict.txt
       since it just prints the prediction to standard output otherwise.'''
    (training, labels) = parseTraining('training_data.txt', '|')
    test = parseTest('testing_data.txt', '|')
    xTrain, xValidate, yTrain, yValidate = cross_validation.train_test_split(training, labels, test_size=0.1)
    # Boosted Decision Trees
    # We train a number of trees of max depth 3 (the default argument). Use
    # 'exponential' to tell the class to use the AdaBoost algoritum.
    boost1 = GradientBoostingClassifier(loss = 'exponential', n_estimators = 200)
    # scores = cross_validation.cross_val_score(boost1, training, labels, cv=5)
    # print("Boost Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # boost1 = boost1.fit(training, labels)

    # AdaBoost blaster
    boost2 = AdaBoostClassifier(n_estimators = 200)
    # scores = cross_validation.cross_val_score(boost2, training, labels, cv=5)
    # print("Ada Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Trains some EXTRA random decision trees.
    rand = ExtraTreesClassifier(n_estimators = 200)
    # scores = cross_validation.cross_val_score(rand, training, labels, cv=5)
    # print("Rand Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # rand = rand.fit(training, labels)

    # Trains a number of bagged deep decision trees with default arguments to
    # RandomForestClassifier except n_estimates = nbagged
    bag = RandomForestClassifier(n_estimators = 200)
    # scores = cross_validation.cross_val_score(bag, training, labels, cv=5)
    # print("Bag Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # bag = bag.fit(training, labels)

    # Trains some decision tree classifiers and picks the best one
    # train_err = []
    # val_err = []
    # ind = 0
    # max_score = 0
    # for i in np.arange(1, 200, 5):
    # # Specify tree params
    #     dTree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = i)
    #     # Calculate Error
    #     scores = cross_validation.cross_val_score(dTree, training, labels, cv = 5)
    #     mean = scores.mean()
    #     if mean > max_score:
    #         max_score = mean
    #         ind = i
    #         print("Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # print("Tree min samples leaf is %f" % ind)
    # bestTree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = ind)
    bestTree = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = 2)

    #SVM
    # ind = 0
    # max_score = 0
    # for i in np.arange(0.1, 1, 0.1):
    #     supp_lin = svm.LinearSVC(C=i, penalty='l1', dual=False)
    #     scores = cross_validation.cross_val_score(supp_lin, training, labels, cv=5)
    #     mean = scores.mean()
    #     if mean > max_score:
    #         max_score = mean
    #         ind = i
    #         print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("Best SVM C is %f" % ind)
    # supp_lin = svm.LinearSVC(C=ind, penalty='l1', dual=False)
    supp_lin = svm.LinearSVC(C=.2, penalty='l1', dual=False)
    #supp_lin = svm.SVC(C=.5, kernel='linear', probability=True, shrinking=True, decision_function_shape='ovr')
    # scores = cross_validation.cross_val_score(supp_lin, training, labels, cv=5)
    # print("Linear SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #SVM
    # ind = 0
    # max_score = 0
    # for i in np.arange(0.1, 1, 0.1):
    #     supp_lin_l2 = svm.LinearSVC(C=i, dual=False)
    #     scores = cross_validation.cross_val_score(supp_lin_l2, training, labels, cv=5)
    #     mean = scores.mean()
    #     if mean > max_score:
    #         max_score = mean
    #         ind = i
    #         print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("Best SVM C is %f" % ind)
    # supp_lin_l2 = svm.LinearSVC(C=ind, dual=False)
    #supp_lin_l2 = svm.LinearSVC(C=.1, dual=False)

    # rfecv = feature_selection.RFECV(estimator=supp_lin_l2, step=1, cv=cross_validation.StratifiedKFold(labels, 2),
    #           scoring='accuracy')
    # rfecv.fit(training, labels)
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()
    # print(rfecv.ranking_)

    multinomBayes = naive_bayes.MultinomialNB(alpha=1)
    # scores = cross_validation.cross_val_score(multinomBayes, training, labels, cv=5)
    # print("Multinom Bayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # anova_filter = preprocessing.RobustScaler()
    # # 2) svm
    # clf = svm.LinearSVC(dual = False)

    # anova_svm = pipeline.make_pipeline(anova_filter, clf)
    # scores = cross_validation.cross_val_score(anova_svm, training, labels, cv=5)
    # print("Annnova SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    


    #=============================================================================================================
    # BAD MODEL SECTION
    #
    #

    # gaussBayes = naive_bayes.GaussianNB()
    # scores = cross_validation.cross_val_score(gaussBayes, training, labels, cv=5)
    # print("Gauss Bayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # bernBayes = naive_bayes.BernoulliNB()
    # scores = cross_validation.cross_val_score(bernBayes, training, labels, cv=5)
    # print("Bern Bayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # # Gauss Bayes Accuracy: 0.59 (+/- 0.02)
    # # Bern Bayes Accuracy: 0.56 (+/- 0.02)


    #SVM gave ~.55
    # ind = 0
    # max_score = 0
    # for i in np.arange(0.1, 1, 0.1):
    #     supp_sig = svm.SVC(C=i, kernel='sigmoid')
    #     scores = cross_validation.cross_val_score(supp_sig, training, labels, cv=5)
    #     mean = scores.mean()
    #     if mean > max_score:
    #         max_score = mean
    #         ind = i
    #         print("Sig SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("Best sigmoid SVM C is %f" % ind)
    # supp_sig = svm.SVC(C=ind, kernel='sigmoid')

    # #SVM gave ~.56
    # ind = 0
    # max_score = 0
    # for i in np.arange(0.1, 1, 0.1):
    #     supp_poly = svm.SVC(C=i, kernel='poly')
    #     scores = cross_validation.cross_val_score(supp_poly, training, labels, cv=5)
    #     mean = scores.mean()
    #     if mean > max_score:
    #         max_score = mean
    #         ind = i
    #         print("Poly SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("Best poly SVM C is %f" % ind)
    # supp_poly = svm.SVC(C=ind, kernel='poly')
    
    #Aggregate the classifiers with soft voting.
    # eclf = 0
    # max_score = 0
    # for i in np.arange(500, 900, 50):
    #     #scaler = preprocessing.RobustScaler(with_centering=False)
    #     clusterer = cluster.FeatureAgglomeration(n_clusters=i)
    #     #decomposer = decomposition.RandomizedPCA(500)
    #     #training = decomposition.RandomizedPCA(500).fit_transform(training)
    #     #training = np.add(np.amin(training), training)
    #     #clf = VotingClassifier(estimators = [('Ada', boost2), ('Bayes', multinomBayes), ('Lin SVM', supp_lin), ('DecisionTree', bestTree), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
    #     #clf = VotingClassifier(estimators = [('Ada', boost2), ('Lin SVM', supp_lin), ('DecisionTree', bestTree), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
    #     clf = VotingClassifier(estimators = [('DTree', bestTree), ('LinearSVM', supp_lin), ('Bayes', multinomBayes), ('AdaBoost', boost2), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
    #     clf = pipeline.make_pipeline(clusterer, clf)

    #     params = {'Bayes__alpha': [0.5, 1, 2, 4], 'LinearSVM__C': [0.01, 0.1, 0.5, 1, 2], 'DTree__min_samples_leaf': [2, 10, 50, 100, 300, 1000]}

    #     print("About to grid")

    #     grid = grid_search.GridSearchCV(estimator=clf, param_grid=params, n_jobs=7, cv=3)
    #     grid = grid.fit(training, labels)

    #     print("Done gridding")
    #     #scores = cross_validation.cross_val_score(grid, training, labels, cv=4)
    #     mean = grid.best_score_
    #     if mean > max_score:
    #         max_score = mean
    #         eclf = grid
    #         print("Best params %s" % eclf.best_params_)
    #         print("Best estimator %s" % eclf.best_estimator_)
    #         print("Accuracy: %0.2f (+/- %0.2f)" % grid.best_score_, 0))

    eclf = 0
    max_score = 0
    for i in np.arange(500, 900, 50):
    #     #scaler = preprocessing.RobustScaler(with_centering=False)
        clusterer = cluster.FeatureAgglomeration(n_clusters=i)
        #decomposer = decomposition.RandomizedPCA(500)
        #training = decomposition.RandomizedPCA(500).fit_transform(training)
        #training = np.add(np.amin(training), training)
        #clf = VotingClassifier(estimators = [('Ada', boost2), ('Bayes', multinomBayes), ('Lin SVM', supp_lin), ('DecisionTree', bestTree), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
        #clf = VotingClassifier(estimators = [('Ada', boost2), ('Lin SVM', supp_lin), ('DecisionTree', bestTree), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
        clf = VotingClassifier(estimators = [('DTree', bestTree), ('LinearSVM', supp_lin), ('Bayes', multinomBayes), ('AdaBoost', boost2), \
            ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'hard')
        clf = pipeline.make_pipeline(clusterer, clf)

        #params = {'Bayes__alpha': [1], 'LinearSVM__C': [0.01, 0.1, 0.5], 'DTree__min_samples_leaf': [2, 10, 50]}

        print("About to grid")

        #grid = grid_search.GridSearchCV(estimator=clf, param_grid=params, n_jobs=7, cv=3)
        #grid = grid.fit(training, labels)

        print("Done gridding")
        #scores = cross_validation.cross_val_score(grid, training, labels, cv=4)
        scores = cross_validation.cross_val_score(clf, training, labels, cv=4)
        #mean = grid.best_score_
        mean = scores.mean()
        if mean > max_score:
            max_score = mean
            #eclf = grid
            eclf = clf
            # print("Best params %s" % eclf.best_params_)
            # print("Best estimator %s" % eclf.best_estimator_)
            # print("Accuracy: %0.2f (+/- %0.2f)" % (eclf.best_score_, 0))
            print("Accuracy: %0.2f (+/- %0.2f)" % (mean, scores.std()))


    # clusterer = cluster.FeatureAgglomeration(n_clusters=700)
    # clf = VotingClassifier(estimators = [('AdaBoost', boost2), ('GradBoost', boost1), ('ExtraRDT', rand), ('BaggedDT', bag)], voting = 'soft')
    # eclf = pipeline.make_pipeline(clusterer, clf)
    # scores = cross_validation.cross_val_score(eclf, training, labels, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    eclf.fit(training, labels)
    predict(eclf, test)

# class Shifter:
#     def __init__(self):
#         print("Shifter made!")
# =======
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
