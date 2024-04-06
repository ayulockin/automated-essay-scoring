import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv")

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.score.values.tolist()
)

print(len(train_df))
print(len(valid_df))

train_df.to_csv("data/train_df.csv", index=False)
valid_df.to_csv("data/valid_df.csv", index=False)
