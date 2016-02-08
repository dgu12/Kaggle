import tensorflow as tf
import skflow

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn import tree, cross_validation, svm, feature_selection, naive_bayes, pipeline, preprocessing, grid_search, cluster, decomposition, metrics
from tensorflow.models.rnn import rnn, rnn_cell
import random

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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__ == '__main__':
    '''This file is meant to be run with something like
       python parse.py > predict.txt
       since it just prints the prediction to standard output otherwise.'''
    (training, labels) = parseTraining('training_data.txt', '|')
    test = parseTest('testing_data.txt', '|')
    xTrain, xValidate, yTrain, yValidate = cross_validation.train_test_split(training, labels, test_size=0.1)

    for i in range(len(xTrain)):
        for j in range(24):
            xTrain[i].append(0)
    for i in range(len(xValidate)):
        for j in range(24):
            xValidate[i].append(0)
    for i in range(len(test)):
        for j in range(24):
            test[i].append(0)

    xTrain = np.matrix(xTrain)
    yTrain2 = np.zeros((len(yTrain), 2))
    yValidate2 = np.zeros((len(yValidate), 2))
    for a in range(len(yTrain)):
        if yTrain[a] == 1:
            yTrain2[a, 1] = 1
        else:
            yTrain2[a, 0] = 1

    for b in range(len(yValidate)):
        if yValidate[b] == 1:
            yValidate2[b, 1] = 1
        else:
            yValidate2[b, 0] = 1

    x = tf.placeholder(tf.float32, [None, 1024])
    W = tf.Variable(tf.zeros([1024, 2]))
    b = tf.Variable(tf.zeros([2]))
    y_ = tf.placeholder(tf.float32, [None, 2])
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 32, 32, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([8*8*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    



    y = tf.nn.softmax(tf.matmul(x, W) + b)
 
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10, 1.0)))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for j in range(10000):
        idx = random.sample(xrange(len(xTrain)), 50)
        batch_xs = np.zeros((50,1024))
        batch_ys = np.zeros((50,2))
        for i in range(50):
            batch_xs[i,:] = xTrain[idx[i],:]
            batch_ys[i,:] = yTrain2[idx[i], :]
        if j%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
            print("step %d, training acc %g" %(j, train_accuracy))
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 0.5})


    print sess.run(accuracy, feed_dict={x:xValidate, y_:yValidate2, keep_prob: 1.0})
    print "Id,Prediction"
    predictions = sess.run(tf.argmax(y_conv,1), feed_dict={x:test, keep_prob:1.0})
    i = 0
    for p in predictions:
        print "%i,%i" % (i + 1, p)
        i = i+1

    