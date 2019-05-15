load("//:databazel.bzl", "train")

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "train",
    deps = [":lib"],
    srcs = ["train.py"],
    data = ["//:mnist_data"],
    args = ["data/mnist.npz", "/Users/df/code/databazel/model.h5py"]
)

train(
    name = "bzl_training",
    deps = [":lib"],
    train_executable = ":train",
    training_data = "//:mnist_data",
    model_output_path = "bzl_model.h5py"
)
