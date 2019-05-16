from argparse import ArgumentParser
import os

import keras.backend as K
from keras.models import load_model

from mnist.lib import load_test_data, prep_x_data, prep_y_data


def main(model_path, data_path, output_dir):
    # Load the model
    model = load_model(model_path)
    # Load the test data
    x_test, y_test = load_test_data(data_path)
    x_test = prep_x_data(x_test)
    y_test = prep_y_data(y_test)
    # Evaluate the model on the test data
    results = model.evaluate(x_test, y_test)
    # Write out the results of evaluation into file(s) in the output dir
    # For now just going to write out some text...
    results_output = os.path.join(output_dir, 'results.txt')
    with open(results_output, 'w') as f:
        # just cause I know that it's coming out as a scalar now
        f.write(str(results))
        f.write('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--data-path')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_dir)
