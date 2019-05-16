load("//:databazel.bzl", "model")
load("//:databazel.bzl", "evaluate")
load("//:databazel.bzl", "model_with_hyperparam_values")
load("//:databazel.bzl", "hyperparam_search")



py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "train",
    deps = [":lib"],
    srcs = ["train.py"],
    # data = ["//:mnist_data"],
    # args = ["data/mnist.npz", "/Users/df/code/databazel/model.h5py"]
)

py_binary(
    name = "evaluate",
    deps = [":lib"],
    srcs = ["evaluate.py"],
)

model_with_hyperparam_values(
    name = "bzl_training",
    deps = [":lib"],
    train_executable = ":train",
    training_data = "//:mnist_data",
    model_output = "bzl_model.h5py",
    hyperparam_values_dict = {
        "dense_size": ["64", "128"]
    }
)

evaluate(
    name = "bzl_eval",
    deps = [":lib"],
    eval_executable = ":evaluate",
    test_data = "//:mnist_data",
    outputs = ["results.json"],
    model = ":bzl_training"
)

hyperparam_search(
    name = "bzl_search",
    deps = [":lib"],
    eval_executable = ":evaluate",
    train_executable = ":train",
    data = "//:mnist_data",
    eval_outputs = ["results.json"],
    hyperparam_values = {"dense_size": ["64", "128"]},
    model_name = "search_model.h5py"
)
