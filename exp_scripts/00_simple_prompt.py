import logging
logging.basicConfig(level=logging.INFO)

import gc
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

import torch
logging.debug(torch.__version__)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

df = pd.read_csv("data/valid_df.csv")
logging.info(f"Total essays: {len(df)}")

# Configs
model_name = "HuggingFaceH4/zephyr-7b-beta"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

run = wandb.init(
    project="score_essay",
    job_type="exp_prompting",
    config={
        "model_name": model_name,
        "bitsandbytes": quantization_config,
    }
)


def get_message(essay):
    return [
        {
            "role": "system",
            "content": "You are a smart essay scoring bot. Grade the score in a range of 1-6 where 1 is the worst and 6 is the best.",
        },
        {
            "role": "user", 
            "content": f"{essay}"
        },
        {
            "role": "user",
            "content": "Give the score as a valid dictionary like this: `{'score': grade}`"
        }
     ]


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
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

generations = []
scores = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    essay = row.full_text.strip(" ")
    message = get_message(essay)
    inputs = tokenizer.apply_chat_template(
        message, 
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_attention_mask=False
    ).to("cuda")
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            ).to("cpu")
    outputs = tokenizer.decode(outputs[0])
    torch.cuda.empty_cache()
    gc.collect()
    # try to parse the score
    try:
        score_dict = eval(outputs.split("<|assistant|>")[-1].split("</s>")[0].strip("\n"))
        score = int(score_dict.get("score"))
    except:
        # Median score is 3
        score = 3

    generations.append(outputs)
    scores.append(score)

logging.info("Scoring Done!")

df = df.loc[:len(generations)-1]
df["generations"] = generations
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
