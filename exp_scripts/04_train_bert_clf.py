import os
os.environ["WANDB_PROJECT"] = "score_essay"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
import logging
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from datasets import load_dataset
from transformers import AutoTokenizer
# from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate


df = pd.read_csv("data/train_df.csv")
class_labels = sorted(df['score'].unique() - 1)
logging.info(f"class_labels: {class_labels}")
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=df['score']-1)
logging.info(f"class_weights: {class_weights}")


model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def preprocess_labels(examples):
    labels = []
    for label in examples["label"]:
        label = label - 1
        # label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=6)
        labels += [label]
    return {"label": labels}

train_ds = load_dataset("csv", data_files="data/train_df.csv").shuffle(seed=42)
valid_ds = load_dataset("csv", data_files="data/valid_df.csv")

train_ds = train_ds.remove_columns("essay_id")
train_ds = train_ds.rename_column("full_text", "text")
train_ds = train_ds.rename_column("score", "label")
valid_ds = valid_ds.remove_columns("essay_id")
valid_ds = valid_ds.rename_column("full_text", "text")
valid_ds = valid_ds.rename_column("score", "label")

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.map(preprocess_labels, batched=True)

tokenized_valid = valid_ds.map(preprocess_function, batched=True)
tokenized_valid = tokenized_valid.map(preprocess_labels, batched=True)

config = DebertaV2Config.from_pretrained(model_name)
print(config)
model = DebertaV2ForSequenceClassification(config=config).from_pretrained(model_name, num_labels=6)
print(model)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="clf",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu=False,
    report_to="wandb",
    logging_steps=10,
    overwrite_output_dir=True,
)


from transformers import Trainer
import torch

class MyTrainer(Trainer):
    def __init__(self, class_weights, label_smoothing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.

            # Changes start here
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logits = outputs['logits']
            loss = self.criterion(logits, inputs['labels'])
            # Changes end here

        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    class_weights=torch.tensor(class_weights, dtype=torch.float32).to("cuda"),
    label_smoothing=0.0,
    model=model,
    args=training_args,
    train_dataset=tokenized_train["train"],
    eval_dataset=tokenized_valid["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()