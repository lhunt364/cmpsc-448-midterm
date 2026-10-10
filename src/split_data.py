import numpy as np
import pandas as pd

# step 2, takes data/examples.csv -> balance each group's size, then split into training, validation, and testing sets -> output to data/split.csv

SEED = 86

df = pd.read_csv("data/examples.csv", keep_default_na=False)
min_n = df["llm_group"].value_counts().min()
df = df.groupby("llm_group").sample(n=min_n, random_state=SEED)
print("balanced each group to", min_n, "samples")
print("total:", len(df))
print()

# examples contains prompts that could be not unique, and the same prompt shouldn't be in different sets, so split by prompts
rng = np.random.default_rng(SEED)
prompts = np.array(df["prompt"].unique())
rng.shuffle(prompts)

# 70% training, 15% validation, 15% testing
n_train = int(0.70 * len(prompts))
n_val = int(0.15 * len(prompts))

split_of = {}
for p in prompts[:n_train]:
    split_of[p] = "train"
for p in prompts[n_train:n_train + n_val]:
    split_of[p] = "val"
for p in prompts[n_train + n_val:]:
    split_of[p] = "test"

df["split"] = df["prompt"].map(split_of)

# check split totals and that prompts were properly separated
print(pd.crosstab(df["llm_group"], df["split"]))
assert df.groupby("prompt")["split"].nunique().max() == 1, "same prompt exists in multiple sets (not good)"
df.to_csv("data/split.csv", index=False)
print("\nsaved data/split.csv")