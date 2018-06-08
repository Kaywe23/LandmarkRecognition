
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
from keras import backend as K
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import sys
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
from multiprocessing import Pool
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

from tensorflow.python.platform import app
from delf import feature_io

import argparse
import os
import sys
import time
from time import *
import io
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io


maindir ='./LandmarkRecognition/'
mainpath = './models/research/delf/delf/python/examples/'
top_model_weights_path = maindir + 'bottleneck_fc_model.h5'
train_data_dir = maindir + 'train_images'
testfile = maindir + 'test/'


def count(dir):
    i = 0
    count = []
    while i <= 14950:
        f = str(i)
        for root, dirs, files in os.walk(dir +'/'+ f):  # loop through startfolders
            for pic in files:
                count.append(pic)

            i += 1


    return len(count)

if os.path.exists('images.txt'):
    os.remove('images.txt')
if os.path.exists('data/lk_features'):
    shutil.rmtree('data/lk_features')
img_width, img_height = 128, 128



nb_train_samples = count(train_data_dir)
#nb_validation_samples = count(validation_data_dir)
epochs = 2
batch_size = 10
cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

def writetolist(picone, pictwo):
    picdir1 = picone
    picdir2 = pictwo
    with open('images.txt', "w") as file:

        file.writelines(picdir1 + '\n')
        file.writelines(picdir2)

def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def createFeatures(listpath, configpath, outputpath):


  # Read list of images.
  #print ('Reading list of images')
  image_paths = _ReadImageList(listpath)
  num_images = len(image_paths)
  #print ('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(configpath, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not os.path.exists(outputpath):
    os.makedirs(outputpath)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)

    with tf.Session() as sess:
      # Initialize variables.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      # Loading model that will be used.
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 config.model_path)
      graph = tf.get_default_graph()
      input_image = graph.get_tensor_by_name('input_image:0')
      input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num = graph.get_tensor_by_name(
          'input_max_feature_num:0')
      boxes = graph.get_tensor_by_name('boxes:0')
      raw_descriptors = graph.get_tensor_by_name('features:0')
      feature_scales = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])

      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
          boxes, raw_descriptors, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = clock()
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          print('Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (clock() - start)
          #print ('Processing image %d out of %d, last %d '
                          #'images took %f seconds', i, num_images,
                          #_STATUS_CHECK_ITERATIONS, elapsed)
          start = clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0] + _DELF_EXT
        out_desc_fullpath = os.path.join(outputpath, out_desc_filename)
        if tf.gfile.Exists(out_desc_fullpath):
          #print('Skipping %s', image_paths[i])
          continue

        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image:
                     im,
                 input_score_threshold:
                     config.delf_local_config.score_threshold,
                 input_image_scales:
                     list(config.image_scales),
                 input_max_feature_num:
                     config.delf_local_config.max_feature_num
             })

        feature_io.WriteToFile(out_desc_fullpath, locations_out,
                               feature_scales_out, descriptors_out,
                               attention_out)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)

_DISTANCE_THRESHOLD = 0.8
noLR =[]

def checklandmark(landmark_features):

    feats =[]
    for root, dirs, files in os.walk(landmark_features):  # loop through startfolders
        for f in files:
            feats.append(f)

    i = iter(feats)
    labeldelf = mainpath + 'data/lk_features/' + i.next()
    print (labeldelf)
    testdelf = mainpath + 'data/lk_features/' + i.next()
    print (testdelf)

    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(labeldelf)
    num_features_1 = locations_1.shape[0]
    #print("Loaded trainimage number of features:",num_features_1)
    locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(testdelf)
    num_features_2 = locations_2.shape[0]
    #print("Loaded testimage number of features",num_features_2)

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC.

    try:
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=1000)

        inlier_idxs = np.nonzero(inliers)[0]
        #print(inlier_idxs)
        #print(len(inlier_idxs))

        return len(inlier_idxs)

    except:

        return 0


def pred():

    with open(maindir + 'submission.csv', 'rb') as inf, open(maindir + 'submission_filled.csv', 'wb') as outf:
        file = open(maindir + 'submission.csv')
        numline = len(file.readlines())
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        generator = train_datagen.flow_from_directory(train_data_dir, batch_size=batch_size)
        label_map = (generator.class_indices)

        num_classes = 14951

        print ('Number of Testimages:', numline)

        r = csv.DictReader(inf)
        w = csv.writer(outf)
        #w.writerows(['id', 'landmarks'])
        #w.writeheader()
        t = 0
        for row in r:
            t+=1
            image_path = maindir + 'test/' + row['id'] + '.jpg'

            if row['landmarks']:
                lr_string = row['landmarks'].split(' ', 1)[0]
                print('Image ', t, 'of', numline, 'with label', lr_string)
                pred_string = row['landmarks'].split(' ', 1)[1]
                pred = float(pred_string)

                if pred >= 0.99:
                    print("{},{}".format(row['id'], row['landmarks']))
                    w.writerow([row['id'], row['landmarks']])

                else:
                    label = lr_string

                    path, dirs, files = next(os.walk(train_data_dir + '/' + label))
                    file_count = len(files)
                    print('file_count in this folder', file_count)

                    comparisons = 0

                    for root, dirs, files in os.walk(train_data_dir + '/' + label + '/'):  # loop through startfolders
                        for labelpic in files:

                            dirpic = train_data_dir + '/' + label + '/' + labelpic
                            print(dirpic)

                            # write the two images to list
                            writetolist(dirpic, image_path)

                            # create Features for train images of classified label
                            createFeatures(mainpath + 'images.txt', mainpath + 'delf_config_example.pbtxt',
                                           mainpath + 'data/lk_features')

                            isLandmark = checklandmark('data/lk_features')
                            comparisons += 1
                            print('Pic:', t, 'Number of Inlier in comparison', comparisons, ':', isLandmark)

                            if isLandmark >= 35:
                                # print('Compared:', comparisons)

                                out = row['landmarks']
                                print("{},{}".format(row['id'], out))
                                w.writerow([row['id'], out])

                                if os.path.exists('images.txt'):
                                    os.remove('images.txt')
                                if os.path.exists('data/lk_features'):
                                    shutil.rmtree('data/lk_features')
                                break




                            else:

                                if comparisons >= 20 or comparisons >= file_count:
                                    # print('Compared:', comparisons)

                                    print("{},{}".format(row['id'], ' '))
                                    w.writerow([row['id'], ' '])
                                    if os.path.exists('images.txt'):
                                        os.remove('images.txt')
                                    if os.path.exists('data/lk_features'):
                                        shutil.rmtree('data/lk_features')
                                    break

                                if os.path.exists('images.txt'):
                                    os.remove('images.txt')
                                if os.path.exists('data/lk_features'):
                                    shutil.rmtree('data/lk_features')

            else:

                    orig = cv2.imread(image_path)
                    image = load_img(image_path, target_size=(128, 128))
                    image = img_to_array(image)

                    # important! otherwise the predictions will be '0'
                    image = image / 255

                    image = np.expand_dims(image, axis=0)

                    # classify landmark
                    base_model = applications.VGG16(weights='imagenet', include_top=False,
                                                    input_shape=(128, 128, 3))

                    top_model = Sequential()
                    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
                    top_model.add(Dense(256, activation='relu'))
                    top_model.add(Dropout(0.5))
                    top_model.add(Dense(14951, activation='softmax'))

                    model = Model(input=base_model.input, output=top_model(base_model.output))
                    model.load_weights(top_model_weights_path)

                    prediction = model.predict(image)

                    class_predicted = prediction.argmax(axis=1)
                    # class_predicted = np.argmax(prediction,axis=1)


                    inID = class_predicted[0]
                    # print inID

                    inv_map = {v: k for k, v in label_map.items()}
                    # print class_dictionary

                    label = inv_map[inID]

                    path, dirs, files = next(os.walk(train_data_dir + '/' + label))
                    file_count = len(files)
                    print('file_count in this folder', file_count)

                    comparisons = 0

                    for root, dirs, files in os.walk(train_data_dir + '/' + label + '/'):  # loop through startfolders
                        for labelpic in files:

                            dirpic = train_data_dir + '/' + label + '/' + labelpic
                            print(dirpic)

                            # write the two images to list
                            writetolist(dirpic, image_path)

                            # create Features for train images of classified label
                            createFeatures(mainpath + 'images.txt', mainpath + 'delf_config_example.pbtxt',
                                           mainpath + 'data/lk_features')

                            isLandmark = checklandmark('data/lk_features')
                            comparisons += 1
                            print('Pic:', t, 'Number of Inlier in comparison', comparisons,':', isLandmark)

                            if isLandmark >= 35:
                                #print('Compared:', comparisons)

                                out = row['landmarks']
                                print("{},{}".format(row['id'], out))
                                w.writerow([row['id'], out])

                                if os.path.exists('images.txt'):
                                    os.remove('images.txt')
                                if os.path.exists('data/lk_features'):
                                    shutil.rmtree('data/lk_features')
                                break




                            else:

                                if comparisons >= 20 or comparisons >= file_count:
                                    #print('Compared:', comparisons)

                                    print("{},{}".format(row['id'], ' '))
                                    w.writerow([row['id'], ' '])
                                    if os.path.exists('images.txt'):
                                        os.remove('images.txt')
                                    if os.path.exists('data/lk_features'):
                                        shutil.rmtree('data/lk_features')
                                    break

                                if os.path.exists('images.txt'):
                                    os.remove('images.txt')
                                if os.path.exists('data/lk_features'):
                                    shutil.rmtree('data/lk_features')



            K.clear_session()

pred()

