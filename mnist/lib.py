import sys

from keras import backend as K
from keras.utils import to_categorical
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28

# Used by prep_y_data - could infer but whatever
num_classes = 10


def load_train_data(path):
    f = np.load(open(path, 'rb'))
    return f['x_train'], f['y_train']


def load_test_data(path):
    f = np.load(open(path, 'rb'))
    return f['x_test'], f['y_test']


def prep_x_data(x):
    if K.image_data_format() == 'channels_first':
        raise Exception('boom')
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x = x.astype('float32')
    x /= 255
    return x


def prep_y_data(y):
    return to_categorical(y, num_classes)


if __name__ == "__main__":
    data = load_train_data(sys.argv[1])
    prepped_data = prep_x_data(data[0])
    print(prepped_data.shape)
