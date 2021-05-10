from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv1').output)

img_path = '../GraphSAINT/data/flickr_images/0464_245884868.jpg' #, '../GraphSAINT/data/flickr_images/0014_1993869440.jpg']
img = image.load_img(img_path, target_size=(224, 224, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(x.shape)

features = model.predict(x)
print(features.shape)

