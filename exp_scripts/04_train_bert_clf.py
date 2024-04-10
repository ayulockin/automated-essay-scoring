import os
os.environ["WANDB_PROJECT"] = "score_essay"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate

model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def preprocess_labels(examples):
    labels = []
    for label in examples["label"]:
        labels += [label -1]
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

config = DistilBertConfig.from_pretrained(model_name)
model = DistilBertForSequenceClassification(config=config).from_pretrained(model_name, num_labels=6)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="clf",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu=False,
    report_to="wandb",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train["train"],
    eval_dataset=tokenized_valid["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()