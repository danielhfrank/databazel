# databazel
![databazel](https://user-images.githubusercontent.com/409199/60114102-f87d7100-9727-11e9-886e-10aeebe437e1.png)

# Background

Computing can be complicated. As programmers, we naturally want to decompose a complex process into simpler steps. but how do those steps fit together? Before long, we always turn to DAGs. When building software, we can express build DAGs in `make`. In the realm of data pipelines, tools like `luigi` and `airflow` similarly create DAGs to describe computations made over data. I thought, "we need to build software to run our data analysis workflows, but we use different tools to describe the DAGs to build the two. Why?". With a little too much time on my hands, I created `databazel` to unify the two.

# What

`databazel` is, at its core, a set of bazel rules that runs a machine learning workflow. If you don't know what bazel is, I don't know why you've come this far. This is a pure proof-of-concept of the background discussion above, simply to see if it could be done. As such, it's not very flexible, but it is cool. Currently, it supports three operations: training a model, evaluating a model, and conducting hyperparameter search.

In addition to the bazel rules, there is an example workflow in the `mnist` directory. `databazel` itself does not implement all of the steps above - rather, it specifies how they fit together and depend on one another. A hypothetical user would be responsible for writing training and evaluation scripts that conform to `databazel`'s expected interfaces.

# Run
To run this example, first run `setup.sh` to prep the data. It's lame, I could make data-fetching happen in bazel, but this isn't my day job.
You can then run model training by running `bazel build --python_top=//:env-databazel-3 //mnist:bzl_training`

> Why `--python_top`?

It's what we have to do to make bazel use the python environment set up for our purposes. It's lame. With a different implementation of training we wouldn't need it

> Why `bazel build`?

You may think of model training as being something that you run. But this is a software build tool, so it thinks that it's building something. If you think of it as building a trained model, you'll be able to live with it.
  
Training a model isn't cool. You know what's cool? Training a bunch of models. Automatically. To do a hyperparameter grid search, do `bazel build --python_top=//:env-databazel-3 //mnist:bzl_search`

This will kick off training runs for each combination of specified hyperparameters, and then a model evaluation run for each of them, and write out all the results.

# Cool

Here are some things that people have commented would be cool, but that I haven't done yet or would only make sense if this were being used for real
* Make use of bazel (remote) cache to re-use artifacts (models, evaluation reports) that have already been generated on previous runs. This actually sorta works.
* Make use of bazel remote execution to parallelize things like hyperparam search
* Encode multiple training epochs in bazel. Resume painlessly, in theory.

# API
Here are the components in a little more detail:

## `model`
* training_data: Bazel label for location of training data. Can point to a file, or an s3 download link, or whatever the hell else bazel can understand - it'll figure out how to pass it to your training script
* train_executable: Your script. better make sure it takes the right inputs and writes out its data where it's supposed to
* model: Name of the file that will be written out. This will be passed to your script, which better write the damn file there.
* hyperparams: Just what it sounds like. all strings all the time, your code can convert values to the types you want

## `evaluate`
* eval_executable: Your evaluation script. Again, this needs to conform to the exact contract that `databazel` expects
* test_data: a reference to data like above. The mnist example is a little weird in that it's one big dataset that has both train and test data in it; you can figure out the details within your script
* model: the bazel label of an instance of `model` above

## `hyperparam_search`
This rule does a lot more heavy lifting itself, constructing a series of `model` and `evaluate` steps on its own. You do not need to define those on your own to use it. Its arguments are basically the union of all the arguments above.
