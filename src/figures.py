import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

# step 5, make report figures using all the jsons in results/ and also results/rq4_stats.csv, save to figures/

LABELS = ["Claude", "GPT", "Llama", "PaLM"]
MODEL_NAMES = ["cnn", "lstm"]
MODES = ["prompt", "output", "both"]
MODE_DISPLAY_NAMES = {"prompt": "Prompt only", "output": "Output only", "both": "Prompt + output"}

# one fixed color per group throughout every figure
GROUP_COLORS = {"Claude": "tab:orange", "GPT": "tab:green", "Llama": "tab:blue", "PaLM": "tab:purple"}
ACCENT_BLUE = "tab:blue" # the thing each figure is about
NEUTRAL_GREY = "darkgray" # everything it is compared against
TEXT_COLOR = "black"
MUTED_TEXT_COLOR = "dimgray"
CHANCE_ACCURACY = 25      # with 4 classes, there is a 25% chance of randomly guessing one right

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c8c7c0", "axes.labelcolor": TEXT_COLOR,
    "xtick.color": MUTED_TEXT_COLOR, "ytick.color": MUTED_TEXT_COLOR,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

os.makedirs("figures", exist_ok=True)


def load_results(model_name, mode, ablation_suffix=""):
    with open(f"results/{model_name}_{mode}{ablation_suffix}.json") as json_file:
        return json.load(json_file)


def recall_per_group(confusion_matrix):
    # diagonal = correct predictions for each true group, row sum = how many of that group there were
    return 100 * confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)


# the 6 main runs keyed by (model_name, mode)
all_results = {(model_name, mode): load_results(model_name, mode) for model_name in MODEL_NAMES for mode in MODES}

# figure 1: test accuracy by model and input mode (RQ 1 and 2)
figure, axis = plt.subplots(figsize=(7, 3.8))
mode_positions = np.arange(len(MODES))   # x position of each group of bars: 0, 1, 2
bar_width = 0.36
for position_offset, model_name, bar_color in [(-bar_width / 2, "cnn", ACCENT_BLUE), (bar_width / 2, "lstm", NEUTRAL_GREY)]:
    accuracies = [100 * all_results[(model_name, mode)]["test_accuracy"] for mode in MODES]
    bars = axis.bar(mode_positions + position_offset, accuracies, bar_width - 0.04, color=bar_color, label=model_name.upper())
    for bar, accuracy in zip(bars, accuracies):
        bar_center = bar.get_x() + bar.get_width() / 2
        axis.text(bar_center, accuracy + 1.5, f"{accuracy:.1f}%", ha="center", fontsize=9, color=TEXT_COLOR)

axis.axhline(CHANCE_ACCURACY, color=MUTED_TEXT_COLOR, linestyle="--", linewidth=1)
axis.text(len(MODES) - 0.5, CHANCE_ACCURACY + 1.5, "chance (25%)", ha="right", fontsize=9, color=MUTED_TEXT_COLOR)
axis.set_xticks(mode_positions, [MODE_DISPLAY_NAMES[mode] for mode in MODES])
axis.set_ylim(0, 100)
axis.set_ylabel("Test accuracy (%)")
axis.legend(frameon=False, loc="upper left")
figure.tight_layout()
axis.set_title("Test accuracy of the CNN and LSTM by input type", fontsize=10, color=TEXT_COLOR)
figure.savefig("figures/fig1_accuracy_by_mode.png")
plt.close(figure)

# figure 2: confusion matrices for the output only runs (each row as percent of that true group)
figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
for axis, model_name in zip(axes, MODEL_NAMES):
    confusion_matrix = np.array(all_results[(model_name, "output")]["confusion_matrix"])
    row_percents = 100 * confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    axis.imshow(row_percents, cmap="Blues", vmin=0, vmax=100)
    for true_index in range(len(LABELS)):
        for predicted_index in range(len(LABELS)):
            percent = row_percents[true_index, predicted_index]
            count = confusion_matrix[true_index, predicted_index]
            axis.text(predicted_index, true_index, f"{percent:.0f}%\n({count})", ha="center", va="center", fontsize=8, color="white" if percent > 55 else TEXT_COLOR)
    axis.set_xticks(range(len(LABELS)), LABELS)
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted group")
    axis.set_ylabel("True group")
    test_accuracy = 100 * all_results[(model_name, "output")]["test_accuracy"]
    axis.set_title(f"{model_name.upper()}, output only ({test_accuracy:.1f}% accuracy)", fontsize=10, color=TEXT_COLOR)
    axis.spines[:].set_visible(False)
figure.suptitle("Test-set confusion matrices for the output-only CNN and LSTM", fontsize=10, color=TEXT_COLOR)
figure.tight_layout()
figure.savefig("figures/fig2_confusion_output.png")
plt.close(figure)


# figure 3: training curves for the output only runs
figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
for axis, model_name in zip(axes, MODEL_NAMES):
    history = pd.DataFrame(all_results[(model_name, "output")]["history"])  # one row per epoch
    axis.plot(history["epoch"], 100 * history["train_accuracy"], color=NEUTRAL_GREY, linewidth=2, label="train")
    axis.plot(history["epoch"], 100 * history["val_accuracy"], color=ACCENT_BLUE, linewidth=2, label="validation")
    best_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    axis.axvline(best_epoch, color=MUTED_TEXT_COLOR, linestyle=":", linewidth=1)
    axis.text(best_epoch - 0.15, 4, f"kept epoch {best_epoch}", ha="right", fontsize=8, color=MUTED_TEXT_COLOR)
    axis.set_title(f"{model_name.upper()}, output only", fontsize=10, color=TEXT_COLOR)
    axis.set_xlabel("Epoch")
    axis.set_ylim(0, 100)
    axis.set_xticks(history["epoch"])
figure.suptitle("Training and validation accuracy per epoch for the output-only CNN and LSTM", fontsize=10, color=TEXT_COLOR)
axes[0].set_ylabel("Accuracy (%)")
axes[1].legend(frameon=False, loc="upper left")
figure.tight_layout()
figure.savefig("figures/fig3_training_curves.png")
plt.close(figure)


# figure 4: RQ4 style characteristics per group
if os.path.exists("results/rq4_stats.csv"):
    style_stats = pd.read_csv("results/rq4_stats.csv", index_col=0).reindex(LABELS)
    panels = [("median_tokens", "Median length (tokens)"),
              ("pct_with_list", "Responses with a list (%)"),
              ("pct_with_bold", "Responses with **bold** (%)"),
              ("pct_mentions_ai_model", 'Mentions "AI" or "language model" (%)')]
    figure, axes = plt.subplots(1, len(panels), figsize=(9.5, 2.9))
    for axis, (column_name, panel_title) in zip(axes, panels):
        group_values = style_stats[column_name]
        bars = axis.bar(LABELS, group_values, color=[GROUP_COLORS[group] for group in LABELS], width=0.65)
        for bar, value in zip(bars, group_values):
            value_label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value * 1.02, value_label, ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
        axis.set_title(panel_title, fontsize=9, color=TEXT_COLOR)
        axis.set_yticks([])  # values are printed on the bars, no y axis
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(0, group_values.max() * 1.18)
    figure.suptitle("Response style statistics for each LLM group", fontsize=10, color=TEXT_COLOR)
    figure.tight_layout()
    figure.savefig("figures/fig4_rq4_style.png")
    plt.close(figure)


# figure 5: RQ4 ablations, each group's recall for the output only CNN
ABLATIONS = [("", "All tokens (full 256)"),
             ("_no_format", "No punctuation or newlines"),
             ("_first_64", "First 64 tokens only"),
             ("_last_64", "Last 64 tokens only")]
available_ablations = [(suffix, display_name) for suffix, display_name in ABLATIONS if os.path.exists(f"results/cnn_output{suffix}.json")]
if len(available_ablations) > 1:
    figure, axis = plt.subplots(figsize=(8.2, 3.8))
    ablation_colors = [ACCENT_BLUE, "dimgray", "gray", "silver"]
    group_width = 0.8 # total width of one group's group of bars
    bar_width = group_width / len(available_ablations)
    group_positions = np.arange(len(LABELS)) # x position of each group's group (0, 1, 2, 3)
    for ablation_index, ((suffix, display_name), bar_color) in enumerate(zip(available_ablations, ablation_colors)):
        run_results = load_results("cnn", "output", suffix)
        recalls = recall_per_group(np.array(run_results["confusion_matrix"]))
        bar_positions = group_positions - group_width / 2 + bar_width * (ablation_index + 0.5)
        bars = axis.bar(bar_positions, recalls, bar_width - 0.03, color=bar_color, label=f"{display_name}: {100 * run_results['test_accuracy']:.1f}% overall")
        for bar, recall in zip(bars, recalls):
            axis.text(bar.get_x() + bar.get_width() / 2, recall + 1.2, f"{recall:.0f}", ha="center", fontsize=7.5, color=TEXT_COLOR)
    axis.axhline(CHANCE_ACCURACY, color=MUTED_TEXT_COLOR, linestyle="--", linewidth=1)
    axis.text(3.45, CHANCE_ACCURACY + 1.5, "chance", ha="right", fontsize=8, color=MUTED_TEXT_COLOR)
    axis.set_title("Per-family test recall of the output-only CNN under each ablation", fontsize=10, color=TEXT_COLOR)
    axis.set_xticks(group_positions, LABELS)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Test recall (%)")
    axis.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    figure.tight_layout()
    figure.savefig("figures/fig5_ablations.png")
    plt.close(figure)

print("saved figures to figures/")