import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Model
import csv
import os
import cv2
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import math
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import imutils
import keras
from pathlib import Path

# dimensions of our images.
img_width, img_height = 128, 128

top_model_weights_path = 'bottleneck_fc_model.h5'
#train_data_dir = './train_images'
validation_data_dir = './valid_images'
def count(dir):
    i = 0
    count = []
    while i <= 14950:
        f = str(i)
        for root, dirs, files in os.walk(dir +'/'+ f):  # loop through startfolders
            for pic in files:
                count.append(pic)

            i += 1

    print len(count)
    return len(count)
#nb_train_samples = count(train_data_dir)
#nb_validation_samples = count(validation_data_dir)
epochs = 2
batch_size = 10



def predict(image_path):
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = train_datagen.flow_from_directory(validation_data_dir, batch_size=batch_size)
    label_map = (generator.class_indices)
    print label_map


    num_classes = 14951

    # add the path to your test image below





    with open('./submission.csv', 'wb') as csvfile:
        newFileWriter = csv.writer(csvfile)
        newFileWriter.writerow(['id', 'landmarks'])



        for root, dirs, files in os.walk(image_path):  # loop through startfolders
            for pic in files:
                # loop folder and convert image
                path = image_path + pic

                orig = cv2.imread(path)
                image = load_img(path, target_size=(128, 128))
                image = img_to_array(image)

                # important! otherwise the predictions will be '0'
                image = image / 255

                image = np.expand_dims(image, axis=0)

                # load the trained lenet convolutional neural network

                model = load_model('/home/kevin/LandmarkRec/lenetmodel.model')

                # classify the input image
                (nolandmark, landmark) = model.predict(image)[0]

                # build the label
                label1 = "landmark" if landmark > nolandmark else "NoLandmark"
                proba = landmark if landmark > nolandmark else nolandmark
                #label1 = "{}: {:.2f}%".format(label, proba * 100)

                if label1 == "NoLandmark":
                    print "{},{}".format(os.path.splitext(pic)[0],' ')

                    newFileWriter.writerow([os.path.splitext(pic)[0], ' '])

                else:


                    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))


                    top_model = Sequential()
                    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
                    top_model.add(Dense(256, activation='relu'))
                    top_model.add(Dropout(0.5))
                    top_model.add(Dense(14951, activation='softmax'))


                    model = Model(input=base_model.input, output=top_model(base_model.output))
                    model.load_weights(top_model_weights_path)

                    prediction = model.predict(image)


                    class_predicted = prediction.argmax(axis=-1)
                    #class_predicted = np.argmax(prediction,axis=1)
                    #print class_predicted

                    inID = class_predicted[0]
                    #print inID

                    inv_map = {v: k for k, v in label_map.items()}
                    #print class_dictionary

                    label = inv_map[inID]


                    score = max(prediction[0])
                    scor = "{:.2f}".format(score)
                    out = str(label) + ' '+ scor


                    if score <= 0.5:
                        print "{},{}".format(os.path.splitext(pic)[0], ' ')

                        newFileWriter.writerow([os.path.splitext(pic)[0], ' '])
                    else:

                        print "{},{}".format(os.path.splitext(pic)[0], out)

                        newFileWriter.writerow([os.path.splitext(pic)[0], out])
                K.clear_session()



predict('./test/')
