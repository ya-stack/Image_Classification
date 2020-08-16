## Importing necessary modules
import tensorflow
import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16

## Using VGG19 as the model to classify the uploaded image
model=VGG16(weights='imagenet')
model.save('vgg16.h5')