import os

import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow as tf

def generate_df(trainval_dir, test_dir, validation_split):
    """
    generates dfs for train, val, test data for creating PerceptionDataGenerator1
    """
    # get test data
    test = []
    for guid in os.listdir(test_dir):
        id_dir = test_dir + '/' + guid
        if os.path.isdir(id_dir):
            for filename in os.listdir(id_dir):
                if filename[5] == 'c': # cloud image proj, pick cloud only
                    test.append(guid + '/' + filename[:4]) # guid + / + 000x

    test = pd.DataFrame(test, columns=['guid/image'])

    ids_labels = pd.read_csv(trainval_dir + '/labels.csv')
    train, val = train_test_split(ids_labels, test_size=validation_split)

    return train, val, test
                    

class PerceptionDataGenerator1(tf.keras.utils.Sequence):
    """
    data generator object for task 1 of perception
    """
    def __init__(self, dir_name, data, image_generator=None, id_col='guid/image', label_col='label', batch_size=32,
                 image_size=(224,224), n_classes=3):
        self.dir_name = dir_name
        self.data = data
        self.id_col = id_col
        self.label_col = label_col
        self.image_generator = image_generator
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_classes = n_classes


    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))


    def __getitem__(self, index):
        begin_idx = index*self.batch_size
        end_idx = min(len(self.data), (index+1)*self.batch_size)
        size = end_idx - begin_idx

        x = np.empty((size, *self.image_size, 3))
        y = np.empty(size, dtype=int)

        for batch_idx, data_idx in enumerate(range(begin_idx, end_idx)):
            x[batch_idx,] = self.__read_data__(self.data[self.id_col].values[data_idx])
            if self.label_col is not None:
                y[batch_idx] = self.data[self.label_col].values[data_idx]


        if self.label_col is None:
            return x
        else:
            return x, tf.keras.utils.to_categorical(y, self.n_classes)


    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)


    def __read_data__(self, id):
        img = cv2.resize(cv2.imread(self.dir_name + '/' + id + '_image.jpg'), self.image_size).astype(np.float32)

        # Subtract mean pixel and multiple by scaling constant using ImageNet mean/scales
        # Reference: https://github.com/shicai/DenseNet-Caffe
        img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
        img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
        img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017

        # if self.label_col is not None:
        #     img = self.image_generator.random_transform(img)

        return img

