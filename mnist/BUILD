load("//:databazel.bzl", "model")
load("//:databazel.bzl", "evaluate")

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

model(
    name = "bzl_training",
    deps = [":lib"],
    train_executable = ":train",
    training_data = "//:mnist_data",
    model = "bzl_model.h5py",
)

evaluate(
    name = "bzl_eval",
    deps = [":lib"],
    eval_executable = ":evaluate",
    test_data = "//:mnist_data",
    outputs = ["results.txt"],
    model = ":bzl_training"
)
