from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

import pandas as pd
import os

feats = np.load('../GraphSAINT/data/flickr/feats.npy')
feat_lookup = np.loadtxt('../GraphSAINT/data/BoW_int.dat')

urls = pd.read_fwf('../GraphSAINT/data/NUS-WIDE-urls.txt')

vgg_f = []

counter = 0
for img_f in feats:
    for idx, f in enumerate(feat_lookup):
        if np.array_equal(img_f, f):
            counter += 1
            # local move
            link = urls.iloc[idx, 0].split()[0]
            #os.system('mv ../' + '/'.join(link.split('/')[2:]) + ' .')
            img_path = '../GraphSAINT/data/flickr_images/' + link.split('/')[-1]
            img = image.load_img(img_path, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            vgg_f.append(x)
            print(counter)
            break

    # only for testing
    if counter == 2:
        break

f = tf.concat(vgg_f, 0)
print(f.shape)

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

features = model.predict(f)
print(features.shape)
features = tf.reshape(features, [counter, np.prod(features.shape)//counter])
print(features.numpy().shape)

with open('vgg_feats.npy', 'wb') as fh:
    np.save(fh, features.numpy())



