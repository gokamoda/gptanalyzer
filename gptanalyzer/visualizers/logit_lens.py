import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import collections as mc
from matplotlib.colors import Normalize
from torchtyping import TensorType
from transformers import AutoTokenizer

from gptanalyzer.modules import LAYER, SEQUENCE, VOCAB


def logit_lens_image(
    tokenizer: AutoTokenizer,
    tokenized_prompt: list[str],
    logits: TensorType[LAYER, SEQUENCE, VOCAB],
    ax: plt.Axes,
    cmap="Blues",
    cbar=True,
    extended=False,
    k=1,
    secondary_yxis=False,
    attention=None,
):
    """Plot the logit lens heatmap as an image.

    Parameters
    ----------
    tokenizer : AutoTokenizer
    tokenized_prompt : list[str]
    logits : TensorType[LAYER, SEQUENCE, VOCAB]
        Logits from analyzers.logit_lens
    ax : plt.Axes
    k : int, optional
        Top k vocabs to print for each patch, by default 1
    """
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_idxs = torch.topk(probs, k=k, dim=-1)

    text = []
    for layer in topk_idxs.squeeze(-1):
        text.append([tokenizer.decode(idx).replace(" ", "_") for idx in layer])

    norm = Normalize(vmin=0, vmax=1)
    sns.heatmap(
        topk_probs.squeeze(-1).cpu().numpy(),
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=text,
        fmt="",
        cbar=cbar,
    )
    # ax.set_ylim(1, len(topk_probs))
    ax.invert_yaxis()

    if extended:
        assert len(topk_probs) % 2 == 1
        minor_ticks = [i * 2 for i in range(len(topk_probs) // 2 + 1)]
        major_ticks = [i * 2 + 1 for i in range(len(topk_probs) // 2)]

        minor_ticklabels = ["E"] + [
            f"{i}F" for i in range(1, len(topk_probs) // 2 + 1)
        ]
        major_ticklabels = [
            f"{i}A" for i in range(1, len(topk_probs) // 2 + 1)
        ]
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, minor=False)
        ax.set_yticklabels(minor_ticklabels, rotation=0, minor=True)
        ax.set_yticklabels(major_ticklabels, rotation=0, minor=False)
        ax.grid(
            which="major",
            axis="y",
            color="black",
            linestyle="-",
            linewidth=0.5,
        )

        if secondary_yxis:
            second_y = ax.secondary_yaxis("right")
            second_y.set_yticks(minor_ticks, minor=True)
            second_y.set_yticks(major_ticks, minor=False)
            second_y.set_yticklabels(minor_ticklabels, rotation=0, minor=True)
            second_y.set_yticklabels(major_ticklabels, rotation=0, minor=False)
            second_y.spines["right"].set_visible(False)
    else:
        ax.set_yticks(range(0, len(topk_probs)))
        ax.set_yticklabels(range(0, len(topk_probs)))

    if attention is not None:
        print(attention.shape)
        lines = []
        alphas = []
        attention = attention.mean(dim=1)
        for layer_idx, layer in enumerate(attention):
            for query_pos, row in enumerate(layer):
                for key_pos, weight in enumerate(row):
                    if key_pos == 0:
                        continue
                    if weight.item() < 0.01:
                        continue
                    query_x = query_pos + 0.5
                    key_x = key_pos + 0.5
                    query_y = layer_idx + 1.5
                    key_y = layer_idx + 0.5
                    alphas.append(weight.item())
                    lines.append([(query_x, query_y), (key_x, key_y)])
        max_alpha = max(alphas)
        alphas = [alpha / max_alpha for alpha in alphas]
        c = [(1, 0, 0, alpha) for alpha in alphas]

        # lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
        # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

        lc = mc.LineCollection(lines, colors=np.array(c), linewidths=2)
        ax.add_collection(lc)

    ax.set_xticklabels(tokenized_prompt)
