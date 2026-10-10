import pandas as pd
from datasets import load_dataset

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

# keep only 1 turn conversations, so each row is exactly one user prompt and one llm output
print("total dataset size prior to filtering:", len(df))
df = df[df["turn"] == 1]
print("size after turn == 1:", len(df))
df = df[df["language"] == "English"] # also filter for only English conversations
print("size after language == English:", len(df))
df = df[ ~df["openai_moderation"].apply(lambda x: x["flagged"]) ] # also conversations were tagged if they were not appropriate, so remove those too
print("size after openai_moderation:", len(df))
print()