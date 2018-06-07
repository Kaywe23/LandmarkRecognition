# LandmarkRecognition

This repository shows my code within the Landmark Recognition Challenge of Kaggle. It supports my blog post on the website of NovaTec GmbH.

## Getting Started

For running this code on your local machine, you have to install Python 2.7 and the framework Keras with TensorFlow backend.
After that you can choose your favorite IDE, for example PyCharm.

### Prerequisites

You get your image training and test data on the Kaggle Website: 
https://www.kaggle.com/google/google-landmarks-dataset

Additionally you should follow the DELF quick start tutorial to get the necessary libraries:
https://github.com/tensorflow/models/blob/master/research/delf/EXTRACTION_MATCHING.md

This repository already contains the transformed DELF Python application to handle the landmark recognition. You have to copy them into your downloaded DELF model directory (/models/research/delf/delf/python/examples) to make it work. 

### Application files

This repository contains the following files:
  - PredictionPipeline.py - full prediction pipeline to classify the landmark
  - extract_features.py - basic DELF feature extraction
  - match_images.py - basic DELF image matching
  - DownloadImages.py - downloads images from url within the csv
  - LRCNN4.py - trains a VGG16 based on the train images
  - PreprocessImg.py - creates train folder for each label and assigns each image
  - class_indices.npy - indices of classes
  - classifyLR.py - classifies the landmark without the DELF step
  - submission.csv - submission output csv file

## Deployment

1. Download all images
2. Preprocess all images
3. Train your CNN VGG16
4. Predict your test images with the prediction pipeline 

## Authors

* **Kevin Widholm** - consultant at NovaTec Consulting GmbH

## Links

Also read my blog post related to this repository: 

## References

"Large-Scale Image Retrieval with Attentive Deep Local Features",
Hyeonwoo Noh, Andre Araujo, Jack Sim, Tobias Weyand, Bohyung Han,
Proc. ICCV'17
