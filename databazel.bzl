"""Minimalist example of a rule that does nothing."""

def _empty_impl(ctx):
    # This function is called when the rule is analyzed.
    # You may use print for debugging.
    print("This rule does nothing")

empty = rule(implementation = _empty_impl)


def _train_impl(ctx):
    args = [ctx.file.training_data.path, ctx.outputs.model_output_path.path]
    
    ctx.actions.run(
        inputs = [ctx.file.training_data],
        outputs = [ctx.outputs.model_output_path],
        arguments = args,
        progress_message = "Running training script",
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
        "model_output_path": attr.output()
    },
)
