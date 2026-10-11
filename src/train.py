import argparse

from data_utils import make_loaders, LABELS
from models import CNN, LSTM
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy
import os
import json

# step 3, using the loaders from data_utils.py and the models from models.py, train and evaluate each type of model with each input mode (prompt, output, or both)

SEED = 86
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def one_epoch(model, loader, loss_fn, optimizer=None):
    # one iteration of a loader, trains if there's an optimizer and evaluates if not
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_predications, all_labels = [], []

    with torch.set_grad_enabled(is_training):
        for token_ids, labels in loader:
            token_ids = token_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            class_scores = model(token_ids) # (batch, num_classes)
            loss = loss_fn(class_scores, labels)
            if is_training:
                optimizer.zero_grad() # clear the previous batch's gradient
                loss.backward() # compute new gradients for each weight
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step() # move each weight downhill according to their gradients

            total_loss += loss.item() * len(labels)
            all_predications += class_scores.argmax(1).tolist()
            all_labels += labels.tolist()

    average_loss = total_loss / len(all_labels)
    return average_loss, all_predications, all_labels


def train_and_evaluate(model_name, mode, epochs, learning_rate=1e-3, ablation="none"):
    torch.manual_seed(SEED)
    loaders, vocab = make_loaders(mode, ablation=ablation)
    if model_name == "cnn":
        model = CNN(len(vocab))
    else:
        model = LSTM(len(vocab))
    model = model.to(DEVICE)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = -1
    best_weights = None
    best_epoch = 0
    history = []

    print(f"model_name: {model_name}, mode: {mode}, device: {DEVICE}")
    for epoch in range(1, epochs + 1):
        train_loss, train_predications, train_labels = one_epoch(model, loaders["train"], loss_function, optimizer)
        val_loss, val_predications, val_labels = one_epoch(model, loaders["val"], loss_function)
        train_accuracy = accuracy_score(train_labels, train_predications)
        val_accuracy = accuracy_score(val_labels, val_predications)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        best_flag = ""
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_flag = " (best)"
        print(f"epoch: {epoch}, train_loss: {train_loss:.3f}, train_acc: {train_accuracy:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_accuracy:.3f}{best_flag}")

    model.load_state_dict(best_weights)
    test_loss, test_predications, test_labels = one_epoch(model, loaders["test"], loss_function)

    results = {
        "model": model_name,
        "mode": mode,
        "ablation": ablation,
        "test_accuracy": accuracy_score(test_labels, test_predications),
        "test_f1": f1_score(test_labels, test_predications, average="macro"),
        "confusion_matrix": confusion_matrix(test_labels, test_predications).tolist(),
        "labels": LABELS,
        "vocab_size": len(vocab),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history
    }
    print(f"test_accuracy: {results['test_accuracy']:.3f}, macro-F1: {results['test_f1']:.3f}")
    print("confusion matrix (row = true, col = predicted):", LABELS)
    for label, row in zip(LABELS, results["confusion_matrix"]):
        print(f"\t{label}: {row}")
    print()

    ablation_suffix = "" if ablation == "none" else f"_{ablation}"
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_{mode}{ablation_suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "lstm", "all"], default="all")
    parser.add_argument("--mode", choices=["prompt", "output", "both", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ablation", choices=["none", "no_format", "first_64", "last_64"], default="none")
    args = parser.parse_args()

    model_names = ["cnn", "lstm"] if args.model == "all" else [args.model]
    modes = ["prompt", "output", "both"] if args.mode == "all" else [args.mode]

    for model_name in model_names:
        for mode in modes:
            train_and_evaluate(model_name, mode, args.epochs, args.learning_rate, args.ablation)