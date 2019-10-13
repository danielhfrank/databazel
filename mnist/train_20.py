from argparse import ArgumentParser
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model

# from mnist.lib import load_train_data, prep_x_data, prep_y_data

# -- copying in from lib, but not using traditional keras APIs


def load_train_data(path):
    f = np.load(open(path, 'rb'))
    return f['x_train'], f['y_train']


# input image dimensions
img_rows, img_cols = 28, 28

# Should be able to infer...
num_classes = 10


def prep_x_data(x):
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x = x.astype('float32')
    x /= 255
    return x


def mk_model(hyperparams):

    # dense_size = 128
    dense_size = int(hyperparams['dense_size'])

    # final_dropout_frac = 0.5
    final_dropout_frac = float(hyperparams['final_dropout_frac'])

    input_shape = (28, 28, 1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(dense_size, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(final_dropout_frac))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))

    model.build()
    return model


def main(data_path, model_output_path, hyperparams):
    x_train, y_train = load_train_data(data_path)
    x_train = prep_x_data(x_train)

    # y_train = prep_y_data(y_train)
    y_train = tf.cast(y_train, tf.float64)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(1000).batch(32)

    from IPython import embed

    # import pdb; pdb.set_trace()
    model = mk_model(hyperparams)
    optimizer = tf.keras.optimizers.Adam()

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    step, loss, accuracy = train(dataset, model, optimizer, compute_loss, compute_accuracy)


    # model = train(x_train, y_train, hyperparams)

    print('Writing entire model (with weights)')
    model.save(model_output_path)

def train_one_step(model, optimizer, x, y, compute_loss, compute_accuracy):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss


@tf.function
def train(train_ds, model, optimizer, compute_loss, compute_accuracy):
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y, compute_loss, compute_accuracy)
        if step % 10 == 0:
            tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
    return step, loss, accuracy




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--model-output-path")
    parser.add_argument("--hyperparams", type=json.loads)
    args = parser.parse_args()
    main(args.data, args.model_output_path, args.hyperparams)
