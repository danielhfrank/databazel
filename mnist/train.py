import sys

from mnist.lib import load_train_data, prep_x_data

def train(x, y):
    pass

def main(path):
    x_train, y_train = load_train_data(path)
    x_train = prep_x_data(x_train)
    # Will need to pass this to keras later on
    input_shape = x_train.shape[1:]
    print input_shape

if __name__ == "__main__":
    main(sys.argv[1])
