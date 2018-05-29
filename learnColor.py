import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
import pickle

#function to get CIFAR images to train filter on
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#gets CIFAR images with unpicklingalong with color to target
currDir = os.getcwd()
train_data = unpickle(currDir + "\\cifar-10-batches-py\\data_batch_1")
#train_data = unpickle("C:\\Code\\python\\TensorFlow\\cifar\\nn\\cifar-10-batches-py\\data_batch_1")
targetIm = np.load("color.npy")

#turns images into useable array
train = np.asarray(train_data[b'data'],dtype=np.uint8)
trainIms = train.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
targetImages = [targetIm for _ in range(1000)]
targetIms = np.stack(targetImages,axis=0)


#turns array names into more conventional variables for equations
plt.ion()
n_observations = 10000
xs = trainIms
ys = targetIms


#creates equation and cost function
X = tf.placeholder(tf.float32,shape=[32,32,3])
Y = tf.placeholder(tf.float32,shape=[32,32,3])

W = tf.Variable(tf.ones([32,32,3]), name='weight')
b = tf.Variable(tf.ones([32,32,3]), name='bias')
Y_pred = tf.add(tf.multiply(X, np.absolute(W)), np.absolute(b))

cost = tf.reduce_sum(tf.pow((Y_pred) - (Y), 2)) / (n_observations - 1)


#creates optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#learning area
epochs = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for epoch_i in range(epochs):
        #train on every member of image array
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = 0
        #assess training with random test image
        randPaul = np.random.randint(1000, size=1)
        training_cost = sess.run(cost, feed_dict={X: xs[randPaul][0], Y: y})
        #break if minimum reached
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
    idx = np.random.randint(1000,size=1)
    

    #adds filtered image to plot
    endPred = tf.add(tf.multiply(tf.cast(xs[idx][0],dtype=tf.float32), np.absolute(W)), np.absolute(b))
    ender = tf.cast(endPred,dtype=tf.int16)
    print(ender.eval())
    fig=plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 2)
    plt.imshow((ender.eval()))


    #adds random original image to plot
    orig = tf.cast(xs[idx][0],dtype=tf.int16)
    original = tf.cast(orig,dtype=tf.int16)
    fig.add_subplot(1, 2, 1)
    plt.imshow(original.eval())

    #instructions for final display
    plt.show()
    plt.waitforbuttonpress()
    