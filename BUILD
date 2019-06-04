
filegroup(
  name = "mnist_data",
  srcs = ["data/mnist.npz"],
  visibility = ["//visibility:public"]
)

py_runtime(
  # clean-ish environment installed locally on galiano
  name = "env-databazel-3",
  files = [x for x in glob(["env-databazel-3/**"]) if ' ' not in x],
  interpreter = "env-databazel-3/bin/python",
  visibility = ["//visibility:public"]
)
