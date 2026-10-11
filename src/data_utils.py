import re
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# step 2.5, convert the output of split_data.py into a DataLoader for each split that iterates over the split's rows, tokenized and encoded based on the training vocabulary

# three special characters, <pad> is used to make every prompt the same length, <unk> means a word that was rare or didn't appear in training data,
# and <sep> will be used to indicate where the prompt ends and output begins when both are being trained on at once
PAD, UNK, SEP = 0, 1, 2
LABELS = ["Claude", "GPT", "Llama", "PaLM"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


# a token is a newline character, a word, or punctuation, no whitespace
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")

def tokenize(text, ablation="none"):
    tokens = TOKEN_RE.findall(text.lower())
    if ablation == "no_format": # remove everything except word tokens
        tokens = [t for t in tokens if re.match(r"\w", t)]
    return tokens


def build_vocab(texts, min_freq=2, max_size=30000, ablation="none"):
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t, ablation)) # tokenize the text, counting how many times each token appears
    vocab = {"<pad>": PAD, "<unk>": UNK, "<sep>": SEP}
    for word, c in counts.most_common(max_size):
        if c < min_freq: # nothing can be learned from tokens that were used once/rarely, so filter them out
            break
        vocab[word] = len(vocab) # assign each token to a unique number
    return vocab


# convert text into the vocab's way of representing it (integers), using <unk> wherever an unknown word is found
def encode(text, vocab, ablation="none"):
    return [vocab.get(token, UNK) for token in tokenize(text, ablation)]


# encodes a row's prompt/output into a fixed-length sequence depending on what data is being trained on (prompt, output, or both)
def make_ids(row, vocab, mode, max_len=256, max_input=64, ablation="none"):
    if mode == "prompt":
        ids = encode(row["prompt"], vocab, ablation)
    elif mode == "output":
        ids = encode(row["output"], vocab, ablation)
    elif mode == "both":
        ids = encode(row["prompt"], vocab, ablation)[:max_input] + [SEP] + encode(row["output"], vocab)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if ablation == "last_64":
        ids = ids[-64:]
    ids = ids[:max_len]
    if ablation == "first_64":
        ids = ids[:64]
    return ids + [PAD] * (max_len - len(ids))

# creates DataLoaders for the training loop to iterate through the encoded data, one loader for each split (training, validation, testing)
def make_loaders(mode, batch_size=64, max_len=256, ablation="none"):
    df = pd.read_csv("data/split.csv", keep_default_na=False)
    train_df = df[df["split"] == "train"]

    # vocabulary comes from training data only
    if mode == "prompt":
        texts = train_df["prompt"]
    elif mode == "output":
        texts = train_df["output"]
    else:
        texts = pd.concat([train_df["prompt"], train_df["output"]])
    vocab = build_vocab(texts, ablation=ablation)

    loaders = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        X = torch.tensor([make_ids(r, vocab, mode, max_len, ablation=ablation) for _, r in part.iterrows()]) # for each row in the split, encode it and fix its size, and make it a tensor, shape is(N=examples in split, max_len=256)
        y = torch.tensor([LABEL_TO_ID[g] for g in part["llm_group"]]) # the label for each row, so shape is an N-length list of labels (labels are Claude = 0, GPT = 1, Llama = 2, PaLM = 3)
        loaders[split] = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=(split == "train"))
    return loaders, vocab


if __name__ == "__main__":
    print(tokenize("Sure! Here's a **list**:\n1. Joe"))

    df = pd.read_csv("data/split.csv", keep_default_na=False)
    lengths = df["output"].apply(lambda t: len(tokenize(t)))
    print("output length in tokens: median", lengths.median())
    print("90th percentile: ", lengths.quantile(0.9))
    print()

    loaders, vocab = make_loaders("output")
    print("vocab length:", len(vocab))
    X, y = next(iter(loaders["train"]))
    print("one batch: X", tuple(X.shape), "\ny", tuple(y.shape))
    print("first example, first 30 ids:", X[0][:30].tolist(), "\nlabel:", y[0].item())