""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
from starter import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

''' data processing '''
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = flattenData(trainData), flattenData(validData), flattenData(testData)
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

# Training Parameters
learning_rate = 0.00001 # for Adam
batch_size = 32
epochs = 50

# Network Parameters
num_input = 784 # img shape: 28*28
num_classes = 10 # 10 classes 'A-J'
dropout = 1 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# convolution and ReLU layer
def conv2d(x, W, b, strides=1):
    # x: input, W: filters with 32 filters, b: biases
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# batch normalization layer
def batchnorm(x):
    axises = np.arange(len(x.shape) - 1)
    axises = [0,1,2]
    batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
    normed = tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, 1e-5)
    return normed

# max pooling
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv_graph(x, weights, biases, dropout):

    # step 1: input 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # step 2&3: Convolution and relu
    conv = conv2d(x, weights['wc1'], biases['bc1'])

    # step 4: Batch normalization (might be wrong)
    conv = batchnorm(conv)

    # step 5: Max pooling
    pool = maxpool2d(conv, k=2)

    # step 6: Flatten layer to fit fully connected layer input
    ft = tf.reshape(pool, [-1, weights['wd1'].get_shape().as_list()[0]])

    # step 7 & 8: Fully connected layer and relu
    fc1 = tf.add(tf.matmul(ft, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # step 9: Fully connected layer with output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 filters, should have the same type as input.
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], 0.0, tf.math.sqrt(2 / (3*3 + 3*3)))),
    # fully connected, 14*14*32 inputs, 784 outputs
    'wd1': tf.Variable(tf.random_normal([14*14*32, 784], 0.0, tf.math.sqrt(2 / (14*14*32 + 784)))),
    # 784 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([784, num_classes], 0.0, tf.math.sqrt(2 / (784 + 10))))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([784])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_graph(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # divide mini-batches
    data_size = trainData.shape[0]
    num_batches_per_epoch = data_size // batch_size
    if data_size % batch_size: num_batches_per_epoch += 1

    # mini-batch
    for i in range(epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            feed_dict = {
                X: trainData[start_index:end_index],
                Y: trainTarget[start_index:end_index],
                keep_prob: dropout
            }
            sess.run(train_op, feed_dict=feed_dict)

        trainloss, trainacc = sess.run([loss_op, accuracy], feed_dict={
            X: trainData,
            Y: trainTarget,
            keep_prob: 1.0
        })

        validloss, validacc = sess.run([loss_op, accuracy], feed_dict={
            X: validData,
            Y: validTarget,
            keep_prob: 1.0
        })
        testloss, testacc = sess.run([loss_op, accuracy], feed_dict={
            X: testData,
            Y: testTarget,
            keep_prob: 1.0
        })

        # shuffle data in each epoch
        randIndx = np.arange(data_size)
        np.random.shuffle(randIndx)
        trainData, trainTarget = trainData[randIndx], trainTarget[randIndx]

        print("epoch: {}, trainloss: {}, validloss: {}, testloss: {}".format(i, trainloss, validloss, testloss))
        print("epoch: {}, trainacc: {}, validacc: {}, testacc: {}".format(i, trainacc, validacc, testacc))