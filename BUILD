
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
  # clean-ish environment installed locally on galiano
  name = "env-databazel-3",
  files = [x for x in glob(["env-databazel-3/**"]) if ' ' not in x],
  interpreter = "env-databazel-3/bin/python",
  visibility = ["//visibility:public"]
)
