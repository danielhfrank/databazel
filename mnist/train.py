from argparse import ArgumentParser
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

from mnist.lib import load_train_data, prep_x_data, prep_y_data

batch_size = 128
# epochs = 12
epochs = 1

# Should be able to infer...
num_classes = 10


def train(x_train, y_train):
    input_shape = x_train.shape[1:]

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=tf.nn.softmax))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=[])  # See comment in evaluate.py on cat accuracy fxn

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    return model


def main(data_path, model_filename, output_dir):
    x_train, y_train = load_train_data(data_path)
    x_train = prep_x_data(x_train)
    # Will need to pass this to keras later on

    y_train = prep_y_data(y_train)

    model = train(x_train, y_train)

    # First save model arch only so that I can read it
    print 'Saving model architecture to json'
    model_arch_output_path = os.path.join(output_dir, 'model.json')
    model_json = model.to_json()
    with open(model_arch_output_path, 'w') as f:
        f.write(model_json)

    print 'Writing entire model (with weights)'
    model_out_path = os.path.join(output_dir, model_filename)
    model.save(model_out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--model-output-filename')
    parser.add_argument('--model-output-dir')
    args = parser.parse_args()
    main(args.data, args.model_output_filename, args.model_output_dir)
