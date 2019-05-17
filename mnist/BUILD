load("//:databazel.bzl", "model")
load("//:databazel.bzl", "evaluate")
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

model(
  name = "bzl_training",
  deps = ["//mnist:lib"],
  training_data = "//:mnist_data",
  train_executable = "//mnist:train",
  model = "//mnist:bzl_model.h5py",
  hyperparams = {"dense_size": "128"},
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
    eval_output = "result.json",
    hyperparam_values = {"dense_size": ["64", "32"]},
    model_name = "search_model.h5py"
)
