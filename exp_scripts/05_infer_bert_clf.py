
import logging
logging.basicConfig(level=logging.INFO)

import gc
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

from transformers import AutoTokenizer
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

df = pd.read_csv("data/valid_df.csv")

model_name = "clf/checkpoint-2598"

run = wandb.init(
    project="score_essay",
    job_type="exp_prompting",
    config={
        "model_name": model_name,
    }
)

def rescale_score(score, min_possible_score, max_possible_score):
    """
    Rescales a score to be between 0 and 1.

    Parameters:
    - score: The score to rescale.
    - min_possible_score: The minimum possible score.
    - max_possible_score: The maximum possible score.

    Returns:
    - Rescaled score between 0 and 1.
    """
    return (score - min_possible_score) / (max_possible_score - min_possible_score)

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = DebertaV2Config.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification(config=config).from_pretrained(model_name, num_labels=6).to("cuda")

scores = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    essay = row.full_text
    inputs = tokenizer(essay, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            logits = model(**inputs).logits.cpu()
            pred = np.argmax(logits[0], axis=0) + 1 # becase they were offset during training

    torch.cuda.empty_cache()
    gc.collect()
    scores.append(int(pred))

df["scores"] = scores

kappa_score = cohen_kappa_score(
    list(df.score.values),
    list(df.scores.values.astype(int)),
    weights="quadratic",
)
kappa_score_rescaled = rescale_score(kappa_score, -1, 1)
logging.info(f"Kappa Score: {kappa_score_rescaled}")

run.log(
    {
        "Predictions": df,
        "kappa_score": kappa_score_rescaled,
    }
)