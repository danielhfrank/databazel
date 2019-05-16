
# Utility functions


def mk_param_summary(hyperparams):
    return '__'.join([hpname + '_' + val for hpname, val in hyperparams.items()])

def insert_param_summary(filename, param_summary):
    basename, extn = filename.rsplit('.', 1)
    return basename + '__' + param_summary + '.' + extn

# Rule definitions

def _model_impl(ctx):
    hyperparams_struct = struct(**ctx.attr.hyperparams)
    args = [
        '--data', ctx.file.training_data.path,
        '--model-output-path', ctx.outputs.model.path,
        '--hyperparams', hyperparams_struct.to_json()
        ]
    
    ctx.actions.run(
        inputs = [ctx.file.training_data],
        outputs = [ctx.outputs.model],
        arguments = args,
        progress_message = "Running training script with args %s" % args,
        executable = ctx.executable.train_executable
    )

model = rule(
    implementation = _model_impl,
    attrs = {
        "deps": attr.label_list(),
        "training_data": attr.label( # TODO turn this into a keyed list or something
            allow_single_file = True,
        ),
        "train_executable": attr.label(
            cfg = "target",
            executable = True
        ),
        "model": attr.output(),
        "hyperparams": attr.string_dict()
    },
)

def model_with_hyperparam_values(name,
                                 deps,
                                 training_data,
                                 train_executable,
                                 model_output,
                                 hyperparam_values_dict):
    # This impl is totally wrong but I'm going to just try to get a macro working
    for hyperparam_name, hyperparam_values in hyperparam_values_dict.items():
        for hyperparam_val in hyperparam_values:
            these_values = {hyperparam_name: hyperparam_val}
            param_summary = mk_param_summary(these_values)
            new_name = name + "__" + param_summary
            model_name, extn = model_output.rsplit('.', 1)
            new_model_name = model_name + '__' + param_summary + '.' + extn
            model(
                name = new_name,
                deps = deps,
                training_data = training_data,
                train_executable = train_executable,
                model = new_model_name,
                hyperparams = these_values
            )


def _evaluate_impl(ctx):
    args = [
        '--data-path', ctx.file.test_data.path,
        '--model-path', ctx.file.model.path,
        '--output-dir', ctx.outputs.outputs[0].dirname
    ]
    ctx.actions.run(
        inputs = [ctx.file.test_data, ctx.file.model],
        outputs = ctx.outputs.outputs,
        arguments = args,
        progress_message = "Running training script with args %s" % args,
        executable = ctx.executable.eval_executable
    )


evaluate = rule(
    implementation = _evaluate_impl,
    attrs = {
        "deps": attr.label_list(),
        "test_data": attr.label(allow_single_file=True),
        "model": attr.label(allow_single_file=True),
        "outputs": attr.output_list(allow_empty=False),
        "eval_executable": attr.label(
            cfg = "target",
            executable = True
        )
    }
)


def _hyperparam_search_impl(ctx):
    # This impl is totally wrong but I'm going to just try to get a macro working
    files_to_build = []
    for hyperparam_name, hyperparam_values in ctx.attr.hyperparam_values.items():
        for hyperparam_val in hyperparam_values:
            # Prep our new unique names for this run
            these_values = {hyperparam_name: hyperparam_val}
            param_summary = mk_param_summary(these_values)
            new_name = ctx.attr.name + "__" + param_summary
            new_model_name = insert_param_summary(ctx.attr.model_name, param_summary)
            # Create the model training instance for this run
            model(
                name = new_model_name,
                deps = ctx.attr.deps,
                training_data = ctx.attr.data,
                train_executable = ctx.attr.train_executable,
                model = new_model_name,
                hyperparams = these_values
            )
            # And then also create an eval instance
            new_eval_outputs = [insert_param_summary(x, param_summary) for x in ctx.attr.eval_outputs]
            eval_task = evaluate(
                name = new_name,
                deps = ctx.attr.deps,
                test_data = ctx.attr.data,
                eval_executable = ctx.attr.eval_executable,
                outputs = ctx.attr.eval_outputs
            )
            # Finally, add the files generated here to the list of default files for the rule
            print("****")
            print(eval_task)
            print(eval_task.default_info)
            files_to_build += eval_task.default_info.files
            
    print(files_to_build)
    fail("boom")


hyperparam_search = rule(
    implementation = _hyperparam_search_impl,
    attrs = {
        "deps": attr.label_list(),
        "data": attr.label(allow_single_file=True),
        "model_name": attr.string(mandatory=True),
        "eval_executable": attr.label(
            cfg = "target",
            executable = True
        ),
        "eval_outputs": attr.string_list(),
        "train_executable": attr.label(
            cfg = "target",
            executable = True
        ),
        "hyperparam_values": attr.string_list_dict(mandatory=True)
    }
)
