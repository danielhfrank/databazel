from argparse import ArgumentParser
import json
import os

from keras.models import load_model

from mnist.lib import load_test_data, prep_x_data, prep_y_data


def main(model_path, data_path, output_file):
    # Load the model
    model = load_model(model_path)
    # Load the test data
    x_test, y_test = load_test_data(data_path)
    x_test = prep_x_data(x_test)
    y_test = prep_y_data(y_test)
    # Evaluate the model on the test data
    results = model.evaluate(x_test, y_test)
    # Write out the results of evaluation into a file
    with open(output_file, 'w') as f:
        result_dict = {mname: mvalue for mname, mvalue in zip(model.metrics_names, results)}
        result_dict['model'] = model_path
        json.dump(result_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--data-path')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_file)
