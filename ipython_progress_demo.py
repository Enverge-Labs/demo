import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell
def __1():
    """
    # IPython HTML Progress Bar Demo

    This notebook demonstrates how to create progress bars similar to those in Hugging Face transformers.
    """

    import time
    from IPython.display import display, HTML

    from transformers.utils.notebook import NotebookProgressCallback
    from transformers.training_args import IntervalStrategy
    return HTML, IntervalStrategy, NotebookProgressCallback, display, time


@app.cell
def __5(HTML, IntervalStrategy, NotebookProgressCallback, display, time):
    """
    ## NotebookProgressCallback from Transformers Demo

    This demonstrates using NotebookProgressCallback from the transformers library
    to simulate progress of an AI training session with a main progress bar and one bar per epoch
    as the metrics table gets populated.
    """


    # Create a mock training state and args for the callback
    class MockTrainingState:
        def __init__(self, max_steps, num_train_epochs):
            self.max_steps = max_steps
            self.num_train_epochs = num_train_epochs
            self.global_step = 0
            self.epoch = 0
            self.log_history = []


    class MockTrainingArgs:
        def __init__(self):
            self.eval_strategy = IntervalStrategy.EPOCH


    class MockTrainingControl:
        def __init__(self):
            pass


    # Function to simulate a training run with NotebookProgressCallback
    def simulate_transformers_training():
        # Training parameters
        num_train_epochs = 5
        steps_per_epoch = 100
        total_steps = num_train_epochs * steps_per_epoch

        # Initialize state, args, and control
        state = MockTrainingState(total_steps, num_train_epochs)
        args = MockTrainingArgs()
        control = MockTrainingControl()

        # Initialize the callback
        callback = NotebookProgressCallback()

        # Start training
        callback.on_train_begin(args, state, control)

        # Simulate epochs
        for epoch in range(1, num_train_epochs + 1):
            state.epoch = epoch

            # Simulate steps within epoch
            for step in range(1, steps_per_epoch + 1):
                state.global_step = (epoch - 1) * steps_per_epoch + step

                # Simulate work
                time.sleep(0.01)  # Reduced sleep time for faster simulation

                # Update progress
                callback.on_step_end(args, state, control)

                # Add log entry periodically
                if step % 20 == 0 or step == steps_per_epoch:
                    # Calculate metrics with simulated improvement
                    training_loss = 0.5 - (0.08 * epoch) + (0.005 * epoch * epoch)
                    training_loss = max(0.05, training_loss)

                    # Add to log history
                    log_entry = {"loss": training_loss, "step": state.global_step}
                    state.log_history.append(log_entry)

                    # Call on_log
                    callback.on_log(args, state, control, logs=log_entry)

            # Simulate evaluation at the end of each epoch
            validation_loss = training_loss + 0.05 - (0.01 * epoch)
            validation_loss = max(0.05, validation_loss)
            accuracy = 0.7 + (0.06 * epoch) - (0.005 * epoch * epoch)
            accuracy = min(0.99, accuracy)

            # Create evaluation metrics
            metrics = {
                "eval_loss": validation_loss,
                "eval_accuracy": accuracy,
                "eval_f1": accuracy - 0.05,
                "eval_runtime": 10.5,
                "eval_samples_per_second": 100,
                "eval_steps_per_second": 10,
                "epoch": epoch,
            }

            # Call on_evaluate
            callback.on_evaluate(args, state, control, metrics=metrics)

        # End training
        callback.on_train_end(args, state, control)


    # Try to run the simulation, with fallback if transformers is not installed
    try:
        simulate_transformers_training()
    except ImportError:
        display(
            HTML("""
        <div style="padding: 10px; background-color: #fff3cd; border-left: 5px solid #ffeeba; margin-bottom: 15px;">
            <h3 style="margin-top: 0;">Transformers Library Not Found</h3>
            <p>This demo requires the Hugging Face Transformers library.</p>
            <p>Please install it with: <code>pip install transformers</code></p>
        </div>
        """)
        )
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
