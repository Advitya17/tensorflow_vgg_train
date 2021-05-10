import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.compat.v1.enable_eager_execution()

import utils
from vgg import vgg16

img1 = utils.load_image("./test_data/dog.png")[:, :, :3]
img2 = utils.load_image("./test_data/quail227.jpg")[:, :, :3]

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch = np.concatenate((batch1, batch2), 0)

with tf.compat.v1.Session() as sess:
    images = tf.compat.v1.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    features = vgg.conv2_1
    print(tf.reshape(features, [2, np.prod(features.shape)//2])) # .numpy().shape

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    utils.print_prob_all(prob, './synset.txt')
