import os
import pandas as pd
from datasets import load_dataset

# step 1, takes dataset from huggingface -> filters into valid examples of one user prompt and one llm response -> output in data/examples.csv

ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
df = ds.to_pandas()
# dataset has 20 models, filter that down to 4 groups, 9 models: GPT = gpt-4 + gpt-3.5-turbo, Claude = claude-v1 + claude-instant-v1, PaLM = palm-2, Llama = vicuna-13b, koala-13b, alpaca-13b, and llama-13b
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print()
print(df.iloc[0])
print()
print("conversation_a of row 0:")
print(df.iloc[0]["conversation_a"])
print()
print("model conversation counts:")
print(pd.concat([df["model_a"], df["model_b"]]).value_counts())
print()

# keep only 1 turn conversations, so each row is exactly one user prompt and two llm outputs, one from each of two models
print("total dataset size prior to filtering:", len(df))
df = df[df["turn"] == 1]
print("size after turn == 1:", len(df))
df = df[df["language"] == "English"] # also filter for only English conversations
print("size after language == English:", len(df))
df = df[ ~df["openai_moderation"].apply(lambda x: x["flagged"]) ] # also conversations were tagged if they were not appropriate, so remove those too (there's no flagged rows in the actual dataset, though)
print("size after openai_moderation:", len(df))
print()

# Map each specific model to its group, GPT, Claude, PaLM, or Llama
GROUPS = {
    "gpt-4": "GPT", "gpt-3.5-turbo": "GPT",
    "claude-v1": "Claude", "claude-instant-v1": "Claude",
    "palm-2": "PaLM",
    "vicuna-13b": "Llama", "koala-13b": "Llama", "alpaca-13b": "Llama", "llama-13b": "Llama"
}

# Each row has two answers (model_a and model_b), so we can get up to two examples depending on which are in groups
rows = []
for _, r in df.iterrows():
    for side in ("a", "b"):
        model = r[f"model_{side}"]
        if model not in GROUPS:
            continue
        conv = r[f"conversation_{side}"]
        rows.append({
            "question_id": r["question_id"],
            "model": model,
            "llm_group": GROUPS[model],
            "prompt": conv[0]["content"],
            "output": conv[1]["content"]
        })

examples = pd.DataFrame(rows)
examples = examples[examples["output"].str.strip() != ""]  # filter out rows with no output

print("total examples:", len(examples))
print(examples["llm_group"].value_counts())
print(examples["model"].value_counts())
os.makedirs("data", exist_ok=True)
examples.to_csv("data/examples.csv", index=False)
print("saved data/examples.csv")