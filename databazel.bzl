
def _train_impl(ctx):
    args = [
        '--data', ctx.file.training_data.path,
        '--model-output-path', ctx.outputs.model.path,
        ]
    
    ctx.actions.run(
        inputs = [ctx.file.training_data],
        outputs = [ctx.outputs.model],
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
    },
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
