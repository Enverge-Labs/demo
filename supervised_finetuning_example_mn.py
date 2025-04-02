import marimo

__generated_with = "0.11.26"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Teacher Model Training

        Code authored by: Shaw Talebi

        [Video](https://youtu.be/4QHg8Ix8WWQ) <br>
        [Blog](https://medium.com/towards-data-science/fine-tuning-bert-for-text-classification-a01f89b179fc) <br>
        Based on example [here](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### imports""")
    return


@app.cell
def _():
    from datasets import load_dataset

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    import evaluate
    import numpy as np
    from transformers import DataCollatorWithPadding
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        evaluate,
        load_dataset,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### load data""")
    return


@app.cell
def _(load_dataset):
    dataset_dict = load_dataset("shawhin/phishing-site-classification")
    return (dataset_dict,)


@app.cell
def _(dataset_dict):
    dataset_dict
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train Teacher Model""")
    return


@app.cell
def _(AutoModelForSequenceClassification, AutoTokenizer):
    # Load model directly
    model_path = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    id2label = {0: "Safe", 1: "Not Safe"}
    label2id = {"Safe": 0, "Not Safe": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                               num_labels=2, 
                                                               id2label=id2label, 
                                                               label2id=label2id,)
    return id2label, label2id, model, model_path, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Freeze base model""")
    return


@app.cell
def _(model):
    for (_name, _param) in model.named_parameters():
        print(_name, _param.requires_grad)
    return


@app.cell
def _(model):
    for (_name, _param) in model.base_model.named_parameters():
        _param.requires_grad = False
    for (_name, _param) in model.base_model.named_parameters():
        if 'pooler' in _name:
            _param.requires_grad = True
    return


@app.cell
def _(model):
    for (_name, _param) in model.named_parameters():
        print(_name, _param.requires_grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Preprocess text""")
    return


@app.cell
def _(tokenizer):
    # define text preprocessing
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    return (preprocess_function,)


@app.cell
def _(dataset_dict, preprocess_function):
    # tokenize all datasetse
    tokenized_data = dataset_dict.map(preprocess_function, batched=True)
    return (tokenized_data,)


@app.cell
def _(DataCollatorWithPadding, tokenizer):
    # create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return (data_collator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Evaluation""")
    return


@app.cell
def _(evaluate, np):
    # load metrics
    accuracy = evaluate.load("accuracy")
    auc_score = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        # get predictions
        predictions, labels = eval_pred

        # apply softmax to get probabilities
        probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        # use probabilities of the positive class for ROC AUC
        positive_class_probs = probabilities[:, 1]
        # compute auc
        auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'],3)

        # predict most probable class
        predicted_classes = np.argmax(predictions, axis=1)
        # compute accuracy
        acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'],3)

        return {"Accuracy": acc, "AUC": auc}
    return accuracy, auc_score, compute_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Train model""")
    return


@app.cell
def _(TrainingArguments):
    # hyperparameters
    lr = 2e-4
    batch_size = 8
    num_epochs = 10

    training_args = TrainingArguments(
        output_dir="bert-phishing-classifier_teacher",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    return batch_size, lr, num_epochs, training_args


@app.cell
def _(
    Trainer,
    compute_metrics,
    data_collator,
    model,
    tokenized_data,
    tokenizer,
    training_args,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Apply Model to Validation Dataset""")
    return


@app.cell
def _(compute_metrics, tokenized_data, trainer):
    # apply model to validation dataset
    predictions = trainer.predict(tokenized_data["validation"])

    # Extract the logits and labels from the predictions object
    logits = predictions.predictions
    labels = predictions.label_ids

    # Use your compute_metrics function
    metrics = compute_metrics((logits, labels))
    print(metrics)
    return labels, logits, metrics, predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Push to hub""")
    return


@app.cell
def save_results_1():
    # push model to hub
    #trainer.save_model()
    return


@app.cell
def inference(model, tokenizer):
    # First, check if CUDA is available
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    # Move your model to the appropriate device
    model = model.to(device)

    # Tokenize the input string
    input_text = "000mclogin.micloud-object-storage-xc-cos-static-web-hosting-qny.s3.us-east.cloud-object-storage.appdomain.cloud"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Map prediction to label
    predicted_label = model.config.id2label[predictions.item()]
    print(f"Predicted label: {predicted_label}")
    return (
        device,
        input_text,
        inputs,
        logits,
        model,
        outputs,
        predicted_label,
        predictions,
        torch,
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
