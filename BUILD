load("//:databazel.bzl", "empty")

filegroup(
  name = "random_data",
  srcs = ["data.csv"]
)

filegroup(
  name = "mnist_data",
  srcs = ["data/mnist.npz"],
  visibility = ["//visibility:public"]
)

py_binary(
  name = "test_script",
  srcs = ["test_script.py"],
  data = [":random_data"]
)


py_binary(
  name = "test_script_deps",
  srcs = ["test_script_deps.py"],
  data = [":random_data"]
)


py_runtime(
  # Minimal bare environment
  name = "env-1",
  # Do this to exclude file names with spaces, which bazel will reject.
  files = [x for x in glob(["env-1/**"]) if ' ' not in x],
  interpreter = "env-1/bin/python",
)


py_runtime(
  # Artisanal DL environment
  name = "deep-learning",
  # Do this to exclude file names with spaces, which bazel will reject.
  files = [x for x in glob(["deep-learning/**"]) if ' ' not in x],
  interpreter = "deep-learning/bin/python",
  visibility = ["//visibility:public"]
)
