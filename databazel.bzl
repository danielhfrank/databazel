"""Minimalist example of a rule that does nothing."""

def _empty_impl(ctx):
    # This function is called when the rule is analyzed.
    # You may use print for debugging.
    print("This rule does nothing")

empty = rule(implementation = _empty_impl)


def _train_impl(ctx):
    args = [
        '--data', ctx.file.training_data.path,
        '--model-output-filename', ctx.outputs.model.basename,
        '--model-output-dir', ctx.outputs.model.dirname
        ]
    
    ctx.actions.run(
        inputs = [ctx.file.training_data],
        outputs = [ctx.outputs.model] + ctx.outputs.additional_outputs,
        arguments = args,
        progress_message = "Running training script with args %s" % args,
        executable = ctx.executable.train_executable
    )

model = rule(
    implementation = _train_impl,
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
        "additional_outputs": attr.output_list()
    },
)
