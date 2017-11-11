from __future__ import print_function

import numpy as np
import cv2
import cPickle,gzip,sys
from keras.utils import np_utils


def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data
def loadbanglalekha():
    data, dataLabel, dataMarking, imageFullName = dataset_load('./FullData.pkl.gz')


    Max=0
    print(imageFullName[0])
    for i in range(len(dataLabel)):
        Max=max(Max,dataLabel[i])


    ''' This Portion is for Labeling and Dividing the dataset. Each sample Contains 1800 Images. Total 84 Samples '''
    X_train = []
    X_test = []
    y_train=[]
    y_test=[]
    from collections import defaultdict
    Dict=defaultdict(lambda:None)
    #
    for i in range(len(dataLabel)):
        if(Dict[dataLabel[i]] is None):
            Dict[dataLabel[i]]=1
        else:
            Dict[dataLabel[i]]=Dict[dataLabel[i]]+1

        if(Dict[dataLabel[i]]>1800):
            Value = data[i]
            NV = cv2.resize(Value, (28, 28))
            X_test.append(NV)
            y_test.append(dataLabel[i])
        else:
            Value=data[i]
            NV = cv2.resize(Value, (28, 28))
            X_train.append(NV)
            y_train.append(dataLabel[i])

    batch_size = 128
    nb_classes = 84
    nb_epoch = 15

    # input image dimensions
    img_rows, img_cols = 28,28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    kernel_size = (5, 5)

    X_train=np.asarray(X_train)
    X_test=np.asarray(X_test)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
    X_test = X_test.reshape(X_test.shape[0],  img_rows, img_cols,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, Y_train), (X_test, Y_test)
