import pandas as pd

# step 2, takes data/examples.csv -> balance each group's size, then split into training and testing sets -> output to data/split.csv

SEED = 86

df = pd.read_csv("data/examples.csv", keep_default_na=False)
min_n = df["llm_group"].value_counts().min()
df = df.groupby("llm_group").sample(n=min_n, random_state=SEED)
print("balanced each group to", min_n, "samples")
print("total:", len(df))
print()

