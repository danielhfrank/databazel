import sys

from keras import backend as K
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28

def load_train_data(path):
    f = np.load(open(path))
    return f['x_train'], f['y_train']

def load_train_data(path):
    f = np.load(open(path))
    return f['x_train'], f['y_train']

def prep_x_data(x):
    if K.image_data_format() == 'channels_first':
        raise Exception('boom')
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x = x.astype('float32')
    x /= 255
    return x

if __name__ == "__main__":
    data = load_train_data(sys.argv[1])
    prepped_data = prep_x_data(data[0])
    print prepped_data.shape
