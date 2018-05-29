#use sepia as rgb(177,104,32)

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL import Image
import pickle

r = int(input("Please enter the red value of your RGB color (between 0 and 255):  "))
g = int(input("Please enter the green value of your RGB color (between 0 and 255):  "))
b = int(input("Please enter the blue value of your RGB color (between 0 and 255):  "))

red = tf.scalar_mul(r,(tf.ones([32,32],dtype=tf.int32)))
blue = tf.scalar_mul(b,(tf.ones([32,32],dtype=tf.int32)))
green = tf.scalar_mul(g,(tf.ones([32,32],dtype=tf.int32)))

rgbImage = tf.transpose(tf.stack([red,green,blue]),[1,2,0])

init = tf.global_variables_initializer()

with tf.Session() as session:
	plt.imshow(rgbImage.eval())
	plt.show()
	np.save("color.npy",rgbImage.eval(),allow_pickle=True)
