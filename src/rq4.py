from data_utils import tokenize, LABELS
import pandas as pd
import json
import math
import os
import re
from collections import Counter

# step 4, analyze llm output diversity and find which tokens are distinctive for each llm group and output in results/

MIN_TOTAL_RESPONSES = 40  # a token appearing in this many responses is 'distinctive'
TOP_TOKENS_PER_GROUP = 12


df = pd.read_csv("data/split.csv", keep_default_na=False)
df["tokens"] = df["output"].apply(tokenize)

# each response's features
df["num_tokens"] = df["tokens"].apply(len)
df["num_words"] = df["output"].apply(lambda t: len(t.split()))
df["num_newlines"] = df["output"].str.count("\n")
df["has_newline"] = df["num_newlines"] > 0
df["has_list"] = df["output"].str.contains(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", regex=True)
df["has_bold"] = df["output"].str.contains(r"\*\*[^*]+\*\*", regex=True)
df["has_code_block"] = df["output"].str.contains("```", regex=False)
df["mentions_ai_model"] = df["output"].str.lower().str.contains(r"language model|as an ai", regex=True)
df["apology"] = df["output"].str.lower().str.contains(r"i am sorry|i'm sorry|i apologize", regex=True)
df["starts_with_sure"] = df["output"].str.strip().str.lower().str.match(r"^(sure|certainly|of course)\b")
df["punct_per_100_tokens"] = df["tokens"].apply(lambda tokens: 100 * sum(1 for t in tokens if not re.match(r"\w", t)) / max(len(tokens), 1))

# lexical diversity -> type to token ratio on the first 100 tokens (only responses with >= 100 tokens, so length doesn't affect the ratio as much)
long_enough = df["num_tokens"] >= 100
df.loc[long_enough, "ttr_first_100"] = df.loc[long_enough, "tokens"].apply(lambda tokens: len(set(tokens[:100])) / 100)

def percent_true(boolean_column):
    return 100 * boolean_column.mean()

def percent_over_256(token_counts):
    return 100 * (token_counts > 256).mean()

summary = df.groupby("llm_group").agg(
    median_tokens=("num_tokens", "median"),
    mean_tokens=("num_tokens", "mean"),
    pct_over_256_tokens=("num_tokens", percent_over_256),
    mean_newlines=("num_newlines", "mean"),
    pct_with_newline=("has_newline", percent_true),
    pct_with_list=("has_list", percent_true),
    pct_with_bold=("has_bold", percent_true),
    pct_with_code_block=("has_code_block", percent_true),
    pct_mentions_ai_model=("mentions_ai_model", percent_true),
    pct_apology=("apology", percent_true),
    pct_starts_sure=("starts_with_sure", percent_true),
    punct_per_100_tokens=("punct_per_100_tokens", "mean"),
    ttr_first_100=("ttr_first_100", "mean")
).reindex(LABELS).round(2)

print(summary.T.to_string())

# most distinctive tokens per llm group
responses_containing = {group: Counter() for group in LABELS}
for group, tokens in zip(df["llm_group"], df["tokens"]):
    responses_containing[group].update(set(tokens))  # how many responses contain the token at least once
responses_per_group = df["llm_group"].value_counts()

distinctive_tokens = {}
for group in LABELS:
    other_families = [other for other in LABELS if other != group]
    other_counts = sum((responses_containing[other] for other in other_families), Counter())
    num_group_responses = responses_per_group[group]
    num_other_responses = sum(responses_per_group[other] for other in other_families)

    scored_tokens = []
    for token, group_count in responses_containing[group].items():
        other_count = other_counts[token]
        if group_count + other_count < MIN_TOTAL_RESPONSES:  # skip rare tokens
            continue
        # smoothed share of responses containing the token
        group_share = (group_count + 1) / (num_group_responses + 2) # + 1 and + 2 stop division or log by zero
        other_share = (other_count + 1) / (num_other_responses + 2)
        log_ratio = math.log(group_share / other_share)  # > 0 means more common in this group
        scored_tokens.append({
            "token": token,
            "pct_in_group": round(100 * group_count / num_group_responses, 1),
            "pct_in_others": round(100 * other_count / num_other_responses, 1),
            "log_ratio": round(log_ratio, 2)
        })
    scored_tokens.sort(key=lambda entry: entry["log_ratio"], reverse=True)
    distinctive_tokens[group] = scored_tokens[:TOP_TOKENS_PER_GROUP]

    token_summaries = [f"{entry['token']!r} ({entry['pct_in_group']}% vs {entry['pct_in_others']}%)" for entry in distinctive_tokens[group]]
    print(f"\n{group}: " + ", ".join(token_summaries))

os.makedirs("results", exist_ok=True)
summary.to_csv("results/rq4_stats.csv")
with open("results/rq4_stats.json", "w") as f:
    json.dump({"summary": summary.reset_index().to_dict(orient="records"), "distinctive_tokens": distinctive_tokens}, f, indent=4)
print("\nsaved results/rq4_stats.csv and results/rq4_stats.json")