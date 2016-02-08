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
    clusterer = cluster.FeatureAgglomeration(n_clusters=900)

    clf = VotingClassifier(estimators = [('DTree', bestTree), ('LinearSVM', supp_lin), ('Bayes', multinomBayes), \
        ('GradBoost', boost1), ('BaggedDT', bag)], voting = 'hard')
    
    

    clf = pipeline.Pipeline([('cluster', clusterer), ('voter', clf)])

    params = {'cluster__n_clusters': [890], 'voter__Bayes__alpha': [1.05], 'voter__LinearSVM__C': [0.35,0.36,0.37,0.38,0.4], 'voter__DTree__min_samples_leaf': [1.5, 2, 2.5]}

    grid = grid_search.GridSearchCV(estimator=clf, iid=False, param_grid=params, n_jobs=-1, cv=3)

    grid = grid.fit(training, labels)

    eclf = grid
                
    print("Best params %s" % eclf.best_params_)
    print("Best estimator %s" % eclf.best_estimator_)
    print("Accuracy: %f" % eclf.best_score_)
    print("Scores: %s" % eclf.grid_scores_)

    predict(eclf, test)
