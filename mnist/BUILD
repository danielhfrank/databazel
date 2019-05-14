py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "train",
    deps = [":lib"],
    srcs = ["train.py"],
    data = ["//:mnist_data"],
    args = ["data/mnist.npz"]
)
