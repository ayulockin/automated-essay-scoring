# let's pull the eval data and parse it.
import re
import ast
import glob
import json
import wandb
import pandas as pd
from sklearn.metrics import cohen_kappa_score


wandb_api = wandb.Api()
artifact = wandb_api.artifact("ayush-thakur/score_essay/run-frgjs3mu-Predictions:v0")
local_path = artifact.download()

data_path = glob.glob(f"{local_path}/*.json")[0]
print(data_path)

with open(data_path, "r") as f:
    data = json.load(f)

columns = data["columns"]
data = data["data"]
df = pd.DataFrame(data, columns=columns)
print(len(df))


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


def parse_response(response):
    match = re.search(r"(?<!`)\{[^}]*\}(?!`)", response)
    grade_dict_str = match.group()
    grade_dict = ast.literal_eval(grade_dict_str)
    return grade_dict.get("score")

scores = []
for idx, row in df.iterrows():
    try:
        score = parse_response(row.generations)
        scores.append(score)
    except:
        scores.append(3)
        print(idx)

df["new_scores"] = scores

kappa_score = cohen_kappa_score(
    list(df.score.values),
    list(df.new_scores.values.astype(int)),
    weights="quadratic",
)
kappa_score_rescaled = rescale_score(kappa_score, -1, 1)
print(kappa_score_rescaled)
