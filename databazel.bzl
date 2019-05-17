
# Utility functions


def mk_param_summary(hyperparams):
    return '__'.join([hpname + '_' + val for hpname, val in hyperparams.items()])

def insert_param_summary(filename, param_summary):
    basename, extn = filename.rsplit('.', 1)
    return basename + '__' + param_summary + '.' + extn

# Internal functions

def _model_internal(data, model_output, hyperparams, ctx):
    hyperparams_struct = struct(**hyperparams)
    args = [
        '--data', data.path,
        '--model-output-path', model_output.path,
        '--hyperparams', hyperparams_struct.to_json()
        ]

    ctx.actions.run(
        inputs = [data],
        outputs = [model_output],
        arguments = args,
        progress_message = "Running training script with args %s" % args,
        executable = ctx.executable.train_executable,
        mnemonic = "MODEL%s" % mk_param_summary(hyperparams).replace('_', '')
    )

def _eval_internal(data, model, output, ctx):
    args = [
        '--data-path', data.path,
        '--model-path', model.path,
        '--output-file', output.path
    ]
    ctx.actions.run(
        inputs = [data, model],
        outputs = [output],
        arguments = args,
        progress_message = "Running training script with args %s" % args,
        executable = ctx.executable.eval_executable,
        mnemonic = "EVAL%s" % model.basename.rsplit('.', 1)[0].replace('_', '')
    )

# Rule definitions

def _model_impl(ctx):
    _model_internal(
        ctx.file.training_data,
        ctx.outputs.model,
        ctx.attr.hyperparams,
        ctx
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


def _evaluate_impl(ctx):
    _eval_internal(
        ctx.file.test_data,
        ctx.file.model,
        ctx.outputs.outputs,
        ctx
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
            # This is going to create the new model file, which we need to declare...
            new_model_file = ctx.actions.declare_file(new_model_name)
            # Create the model training instance for this run
            _model_internal(
                data = ctx.file.data,
                model_output = new_model_file,
                hyperparams = these_values,
                ctx = ctx
            )

            # And then also create an eval instance
            new_eval_output = ctx.actions.declare_file(
                    insert_param_summary(ctx.attr.eval_output, param_summary)
                 ) 
            _eval_internal(
                data = ctx.file.data,
                model = new_model_file,
                output = new_eval_output,
                ctx = ctx
            )
            # Finally, add the files generated here to the list of default files for the rule
            files_to_build.append(new_eval_output)
            
    return DefaultInfo(files=depset(files_to_build))


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
        "eval_output": attr.string(),
        "train_executable": attr.label(
            cfg = "target",
            executable = True
        ),
        "hyperparam_values": attr.string_list_dict(mandatory=True)
    }
)
