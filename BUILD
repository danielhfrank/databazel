filegroup(
  name = "data",
  srcs = ["data.csv"]
)

py_binary(
  name = "test_script",
  srcs = ["test_script.py"],
  data = [":data"]
)
