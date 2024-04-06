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

SYSTEM_PROMPT = """
After reading each essay, assign a holistic score based on the rubric below.
For the following evaluations you will need to use a grading scale between 1 (minimum) and 6 (maximum).
As with the analytical rating form, the distance between each grade (e.g., 1-2, 3-4, 4-5) should be considered equal.

SCORE OF 6: An essay in this category demonstrates clear and consistent mastery, although it may have a few minor errors. A typical essay effectively and insightfully develops a point of view on the issue and demonstrates outstanding critical thinking; the essay uses clearly appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is well organized and clearly focused, demonstrating clear coherence and smooth progression of ideas; the essay exhibits skillful use of language, using a varied, accurate, and apt vocabulary and demonstrates meaningful variety in sentence structure; the essay is free of most errors in grammar, usage, and mechanics.
SCORE OF 5: An essay in this category demonstrates reasonably consistent mastery, although it will have occasional errors or lapses in quality. A typical essay effectively develops a point of view on the issue and demonstrates strong critical thinking; the essay generally using appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is well organized and focused, demonstrating coherence and progression of ideas; the essay exhibits facility in the use of language, using appropriate vocabulary demonstrates variety in sentence structure; the essay is generally free of most errors in grammar, usage, and mechanics.
SCORE OF 4: An essay in this category demonstrates adequate mastery, although it will have lapses in quality. A typical essay develops a point of view on the issue and demonstrates competent critical thinking; the essay using adequate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is generally organized and focused, demonstrating some coherence and progression of ideas exhibits adequate; the essay may demonstrate inconsistent facility in the use of language, using generally appropriate vocabulary demonstrates some variety in sentence structure; the essay may have some errors in grammar, usage, and mechanics.
SCORE OF 3: An essay in this category demonstrates developing mastery, and is marked by ONE OR MORE of the following weaknesses: develops a point of view on the issue, demonstrating some critical thinking, but may do so inconsistently or use inadequate examples, reasons, or other evidence taken from the source texts to support its position; the essay is limited in its organization or focus, or may demonstrate some lapses in coherence or progression of ideas displays; the essay may demonstrate facility in the use of language, but sometimes uses weak vocabulary or inappropriate word choice and/or lacks variety or demonstrates problems in sentence structure; the essay may contain an accumulation of errors in grammar, usage, and mechanics.
SCORE OF 2: An essay in this category demonstrates little mastery, and is flawed by ONE OR MORE of the following weaknesses: develops a point of view on the issue that is vague or seriously limited, and demonstrates weak critical thinking; the essay provides inappropriate or insufficient examples, reasons, or other evidence taken from the source text to support its position; the essay is poorly organized and/or focused, or demonstrates serious problems with coherence or progression of ideas; the essay displays very little facility in the use of language, using very limited vocabulary or incorrect word choice and/or demonstrates frequent problems in sentence structure; the essay contains errors in grammar, usage, and mechanics so serious that
meaning is somewhat obscured.
SCORE OF 1: An essay in this category demonstrates very little or no mastery, and is severely flawed by ONE OR MORE of the following weaknesses: develops no viable point of view on the issue, or provides little or no evidence to support its position; the essay is disorganized or unfocused, resulting in a disjointed or incoherent essay; the essay displays fundamental errors in vocabulary and/or demonstrates severe flaws in sentence structure; the essay contains pervasive errors in grammar, usage, or mechanics that persistently interfere with meaning.
"""


def get_message(essay):
    return [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}",
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
