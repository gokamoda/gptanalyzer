from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from gptanalyzer.models import MyGPT2LMHeadModel, load_model


def weight_heatmap(
    weight: torch.Tensor,
    ax: plt.Axes,
    cbar_ax: plt.Axes,
    cmap: str,
    normalize: plt.Normalize,
) -> None:
    """Draw heatmap of tensor weight.

    Parameters
    ----------
    weight : torch.Tensor
    ax : plt.Axes
    cbar_ax : plt.Axes
    cmap : str
    normalize : plt.Normalize
    """
    weight_numpy = weight.numpy()
    sns.heatmap(
        weight_numpy, ax=ax, cmap=cmap, norm=normalize, cbar_ax=cbar_ax
    )


def attn_qk_norm(model: MyGPT2LMHeadModel, save_dir: str):
    """Draw attention weight and axis-wise norm for each layer/head.

    Parameters
    ----------
    model : MyGPT2LMHeadModel
    save_dir : str
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_layers = model.transformer.config.n_layer
    num_heads = model.transformer.config.n_head

    for layer in tqdm(range(num_layers), leave=False):
        w = model.transformer.h[layer].attn.wqkh.detach().cpu()
        b = model.transformer.h[layer].attn.bqwkh.detach().cpu()
        for head in tqdm(range(num_heads), leave=False):
            save_path = save_dir.joinpath(f"layer_{layer}_qk_head{head}.png")
            # if save_path.exists():
            #     print(f"Skip {save_path}")
            #     continue

            fig = plt.figure(figsize=(20, 20))
            axes: dict[str, plt.Axes] = {
                "heatmap": fig.add_axes([0.21, 0.1, 0.56, 0.56]),  # center
                "bk": fig.add_axes([0.21, 0.77, 0.56, 0.05]),  # top
                "mean_out": fig.add_axes([0.21, 0.69, 0.56, 0.05]),  # top
                "colormap3": fig.add_axes(
                    [0.13, 0.77, 0.04, 0.05]
                ),  # top left
                "colormap2": fig.add_axes(
                    [0.13, 0.69, 0.04, 0.05]
                ),  # top left
                # "bq": fig.add_axes([0.05, 0.1, 0.05, 0.56]),  # left
                "mean_in": fig.add_axes([0.13, 0.1, 0.05, 0.56]),  # left
                "colormap": fig.add_axes([0.78, 0.1, 0.02, 0.56]),  # right
            }

            weight = w[head]
            bias = b[head]

            weight_in_norm = weight.norm(dim=1)  # norm by row
            weight_out_norm = weight.norm(dim=0)  # norm by column

            vmax = max(weight_in_norm.max(), weight_out_norm.max())

            weight_heatmap(
                weight_in_norm.unsqueeze(1),
                ax=axes["mean_in"],
                cbar_ax=axes["colormap2"],
                cmap="Blues",
                normalize=mcolors.Normalize(vmax=vmax),
            )

            weight_heatmap(
                weight_out_norm.unsqueeze(0),
                ax=axes["mean_out"],
                cbar_ax=axes["colormap2"],
                cmap="Blues",
                normalize=mcolors.Normalize(vmax=vmax),
            )

            weight_heatmap(
                weight=weight,
                ax=axes["heatmap"],
                cbar_ax=axes["colormap"],
                cmap="RdBu",
                normalize=mcolors.TwoSlopeNorm(
                    vcenter=0, vmin=weight.min(), vmax=weight.max()
                ),
            )

            weight_heatmap(
                weight=bias.unsqueeze(0),
                ax=axes["bk"],
                cbar_ax=axes["colormap3"],
                cmap="RdBu",
                normalize=mcolors.TwoSlopeNorm(
                    vcenter=0, vmin=bias.min(), vmax=bias.max()
                ),
            )

            k = 15

            k_top = weight_out_norm.topk(k).indices.tolist()
            axes["heatmap"].set_xticks(k_top)
            axes["heatmap"].xaxis.tick_top()
            axes["heatmap"].set_xticks(k_top)
            axes["heatmap"].set_xticklabels([])
            axes["mean_out"].set_xticks(k_top)
            axes["mean_out"].set_xticklabels(k_top, rotation=90)
            axes["mean_out"].set_yticks([])
            axes["mean_out"].set_yticklabels([])

            q_top = weight_in_norm.topk(k).indices.tolist()
            axes["heatmap"].set_yticks(q_top)
            axes["heatmap"].set_yticklabels(q_top)
            axes["mean_in"].set_yticks(q_top)
            axes["mean_in"].set_yticklabels([])
            axes["mean_in"].yaxis.tick_right()
            axes["mean_in"].set_xticks([])

            bias_top = bias.topk(k).indices.tolist()
            bias_bottom = bias.topk(k, largest=False).indices.tolist()
            second_x = axes["bk"].secondary_xaxis("top")
            second_x.set_xticks(bias_top)
            second_x.set_xticklabels(bias_top, rotation=90)
            second_x.spines["top"].set_visible(False)
            axes["bk"].set_xticks(bias_bottom)
            axes["bk"].set_xticklabels(bias_bottom, rotation=90)
            axes["bk"].set_yticklabels(["B_q W_k"])

            fig.savefig(save_path, dpi=500, bbox_inches="tight")
            plt.close(fig)


def attn_weight_norm(model: MyGPT2LMHeadModel, save_dir: str):
    """Draw attention weight and axis-wise norm for each layer/head.

    Parameters
    ----------
    model : MyGPT2LMHeadModel
    save_dir : str
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_layers = model.transformer.config.n_layer
    num_heads = model.transformer.config.n_head

    for layer in tqdm(range(num_layers), leave=False):
        for head in tqdm(range(num_heads), leave=False):
            save_path = save_dir.joinpath(f"layer_{layer}_head_{head}.png")
            if save_path.exists():
                print(f"Skip {save_path}")
                continue

            fig = plt.figure(figsize=(20, 20))
            axes: dict[str, plt.Axes] = {
                "heatmap": fig.add_axes([0.13, 0.1, 0.64, 0.64]),  # center
                "mean_out": fig.add_axes([0.13, 0.77, 0.64, 0.05]),  # top
                "colormap2": fig.add_axes(
                    [0.05, 0.77, 0.04, 0.05]
                ),  # top left
                "mean_in": fig.add_axes([0.05, 0.1, 0.05, 0.64]),  # left
                "colormap": fig.add_axes([0.78, 0.1, 0.02, 0.64]),  # right
            }

            weight: torch.Tensor = (
                model.transformer.h[layer].attn.wvo[head].detach().cpu()
            )

            weight_in_norm = weight.norm(dim=1)  # norm by row
            weight_out_norm = weight.norm(dim=0)  # norm by column

            vmax = max(weight_in_norm.max(), weight_out_norm.max())

            weight_heatmap(
                weight_in_norm.unsqueeze(1),
                ax=axes["mean_in"],
                cbar_ax=axes["colormap2"],
                cmap="Blues",
                normalize=mcolors.Normalize(vmax=vmax),
            )

            weight_heatmap(
                weight_out_norm.unsqueeze(0),
                ax=axes["mean_out"],
                cbar_ax=axes["colormap2"],
                cmap="Blues",
                normalize=mcolors.Normalize(vmax=vmax),
            )

            weight_heatmap(
                weight=weight,
                ax=axes["heatmap"],
                cbar_ax=axes["colormap"],
                cmap="RdBu",
                normalize=mcolors.TwoSlopeNorm(
                    vcenter=0, vmin=weight.min(), vmax=weight.max()
                ),
            )

            k = 15

            out_top = weight_out_norm.topk(k).indices.tolist()
            axes["heatmap"].set_xticks(out_top)
            axes["heatmap"].xaxis.tick_top()
            axes["heatmap"].set_xticks(out_top)
            axes["heatmap"].set_xticklabels([])
            axes["mean_out"].set_xticks(out_top)
            axes["mean_out"].set_xticklabels(out_top, rotation=90)
            axes["mean_out"].set_yticks([])
            axes["mean_out"].set_yticklabels([])

            in_bottom = weight_in_norm.topk(k, largest=False).indices.tolist()
            axes["heatmap"].set_yticks(in_bottom)
            axes["heatmap"].set_yticklabels(in_bottom)
            axes["mean_in"].set_yticks(in_bottom)
            axes["mean_in"].set_yticklabels([])
            axes["mean_in"].yaxis.tick_right()
            axes["mean_in"].set_xticks([])

            fig.savefig(save_path, dpi=500, bbox_inches="tight")
            plt.close(fig)


def weight_norm(model: MyGPT2LMHeadModel, save_dir: str):
    """Draw weight norm for each layer.

    Parameters
    ----------
    model : MyGPT2LMHeadModel
    save_dir : str
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    k = 15

    num_layers = model.transformer.config.n_layer

    for layer in tqdm(range(num_layers), leave=False):
        save_path = save_dir.joinpath(f"layer_{layer}.png")
        # if save_path.exists():
        #     print(f"Skip {save_path}")
        #     continue
        fig = plt.figure(figsize=(20, 14))
        axes: dict[str, plt.Axes] = {
            "emb": fig.add_axes([0.05, 0.03, 0.88, 0.06]),
            "emb_cbar": fig.add_axes([0.95, 0.03, 0.02, 0.06]),
            "qi": fig.add_axes([0.05, 0.15, 0.88, 0.06]),
            "qi_cbar": fig.add_axes([0.95, 0.15, 0.02, 0.06]),
            "ki": fig.add_axes([0.05, 0.27, 0.88, 0.06]),
            "ki_cbar": fig.add_axes([0.95, 0.27, 0.02, 0.06]),
            "vi": fig.add_axes([0.05, 0.39, 0.88, 0.06]),
            "vi_cbar": fig.add_axes([0.95, 0.39, 0.02, 0.06]),
            "vo": fig.add_axes([0.05, 0.51, 0.88, 0.06]),
            "vo_cbar": fig.add_axes([0.95, 0.51, 0.02, 0.06]),
            "ff_i": fig.add_axes([0.05, 0.63, 0.88, 0.06]),
            "ff_i_cbar": fig.add_axes([0.95, 0.63, 0.02, 0.06]),
            "ff_o": fig.add_axes([0.05, 0.75, 0.88, 0.06]),
            "ff_o_cbar": fig.add_axes([0.95, 0.75, 0.02, 0.06]),
            "uemb": fig.add_axes([0.05, 0.89, 0.88, 0.06]),
            "uemb_cbar": fig.add_axes([0.95, 0.89, 0.02, 0.06]),
        }

        def weight_heatmap_row(wight, ax, cbar_ax, cmap, normalize, y_label):
            weight_heatmap(
                wight,
                ax=ax,
                cbar_ax=cbar_ax,
                cmap=cmap,
                normalize=normalize,
            )
            topk = wight.squeeze().topk(k).indices.tolist()
            bottomk = wight.squeeze().topk(k, largest=False).indices.tolist()
            ax.set_xticks(bottomk)
            ax.set_xticklabels(bottomk, rotation=90)
            second_x = ax.secondary_xaxis("top")
            second_x.set_xticks(topk)
            second_x.set_xticklabels(topk, rotation=90)
            second_x.spines["top"].set_visible(False)
            ax.set_yticklabels([y_label])

        emb = model.transformer.wte.weight.detach().cpu()
        emb_norm = emb.norm(dim=0)  # norm by dim
        weight_heatmap_row(
            emb_norm.unsqueeze(0),
            ax=axes["emb"],
            cbar_ax=axes["emb_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=emb_norm.max()),
            y_label="E",
        )

        q = (
            model.transformer.h[layer]
            .attn.c_attn.weight[:, : model.config.n_embd]
            .detach()
            .cpu()
        )
        q_i = q.norm(dim=1)  # norm by row
        weight_heatmap_row(
            q_i.unsqueeze(0),
            ax=axes["qi"],
            cbar_ax=axes["qi_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=q_i.max()),
            y_label="Q_in",
        )

        k = (
            model.transformer.h[layer]
            .attn.c_attn.weight[
                :, model.config.n_embd : model.config.n_embd * 2
            ]
            .detach()
            .cpu()
        )
        k_i = k.norm(dim=1)  # norm by row
        weight_heatmap_row(
            k_i.unsqueeze(0),
            ax=axes["ki"],
            cbar_ax=axes["ki_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=k_i.max()),
            y_label="K_in",
        )

        v = model.transformer.h[layer].attn.wvo.detach().cpu().mean(dim=0)
        v_i = v.norm(dim=1)  # norm by row
        weight_heatmap_row(
            v_i.unsqueeze(0),
            ax=axes["vi"],
            cbar_ax=axes["vi_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=v_i.max()),
            y_label="V_in",
        )

        att_o = v.norm(dim=0)
        weight_heatmap_row(
            att_o.unsqueeze(0),
            ax=axes["vo"],
            cbar_ax=axes["vo_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=att_o.max()),
            y_label="V_out",
        )

        ff_i = model.transformer.h[layer].mlp.c_fc.weight.detach().cpu()
        ff_i_norm = ff_i.norm(dim=1)
        weight_heatmap_row(
            ff_i_norm.unsqueeze(0),
            ax=axes["ff_i"],
            cbar_ax=axes["ff_i_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=ff_i_norm.max()),
            y_label="FF_in",
        )

        ff_o = model.transformer.h[layer].mlp.c_proj.weight.detach().cpu()
        ff_o_norm = ff_o.norm(dim=0)
        weight_heatmap_row(
            ff_o_norm.unsqueeze(0),
            ax=axes["ff_o"],
            cbar_ax=axes["ff_o_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=ff_o_norm.max()),
            y_label="FF_out",
        )

        uemb = model.lm_head.weight.detach().cpu()
        uemb_norm = uemb.norm(dim=0)
        weight_heatmap_row(
            uemb_norm.unsqueeze(0),
            ax=axes["uemb"],
            cbar_ax=axes["uemb_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=uemb_norm.max()),
            y_label="U",
        )

        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.close(fig)


def attn_qkv_in_norm(model, save_dir):
    """Draw dim norm of Q, K, V for each layer.

    Parameters
    ----------
    model : _type_
        _description_
    save_dir : _type_
        _description_
    """
    n_layer = model.transformer.config.n_layer
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    k = 15

    for layer in range(n_layer):
        wq = (
            model.transformer.h[layer]
            .attn.c_attn.weight[:, : model.config.n_embd]
            .detach()
            .cpu()
        )
        wk = (
            model.transformer.h[layer]
            .attn.c_attn.weight[
                :, model.config.n_embd : model.config.n_embd * 2
            ]
            .detach()
            .cpu()
        )
        wv = model.transformer.h[layer].attn.wvo.detach().cpu().mean(dim=0)

        wq_norm = wq.norm(dim=1)
        wk_norm = wk.norm(dim=1)
        wv_norm = wv.norm(dim=1)

        fig = plt.figure(figsize=(20, 10))
        axes: dict[str, plt.Axes] = {
            "q": fig.add_axes([0.05, 0.02, 0.88, 0.10]),
            "q_cbar": fig.add_axes([0.95, 0.02, 0.02, 0.10]),
            "k": fig.add_axes([0.05, 0.24, 0.88, 0.10]),
            "k_cbar": fig.add_axes([0.95, 0.24, 0.02, 0.10]),
            "v": fig.add_axes([0.05, 0.46, 0.88, 0.14]),
            "v_cbar": fig.add_axes([0.95, 0.46, 0.02, 0.10]),
        }

        weight_heatmap(
            wq_norm.unsqueeze(0),
            ax=axes["q"],
            cbar_ax=axes["q_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=wq_norm.max()),
        )
        q_bottom = wq_norm.topk(k, largest=False).indices.tolist()
        axes["q"].set_xticks(q_bottom)
        axes["q"].set_xticklabels(q_bottom, rotation=90)
        axes["q"].set_yticklabels(["Q"])

        q_top = wq_norm.topk(k).indices.tolist()
        second_qx = axes["q"].secondary_xaxis("top")
        second_qx.set_xticks(q_top)
        second_qx.set_xticklabels(q_top, rotation=90)

        weight_heatmap(
            wk_norm.unsqueeze(0),
            ax=axes["k"],
            cbar_ax=axes["k_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=wk_norm.max()),
        )
        k_bottom = wk_norm.topk(k, largest=False).indices.tolist()
        axes["k"].set_xticks(k_bottom)
        axes["k"].set_xticklabels(k_bottom, rotation=90)
        axes["k"].set_yticklabels(["K"])

        k_top = wk_norm.topk(k).indices.tolist()
        second_kx = axes["k"].secondary_xaxis("top")
        second_kx.set_xticks(k_top)
        second_kx.set_xticklabels(k_top, rotation=90)

        weight_heatmap(
            wv_norm.unsqueeze(0),
            ax=axes["v"],
            cbar_ax=axes["v_cbar"],
            cmap="Blues",
            normalize=mcolors.Normalize(vmax=wv_norm.max()),
        )
        v_bottom = wv_norm.topk(k, largest=False).indices.tolist()
        axes["v"].set_xticks(v_bottom)
        axes["v"].set_xticklabels(v_bottom, rotation=90)
        axes["v"].set_yticklabels(["V"])

        v_top = wv_norm.topk(k).indices.tolist()
        second_vx = axes["v"].secondary_xaxis("top")
        second_vx.set_xticks(v_top)
        second_vx.set_xticklabels(v_top, rotation=90)

        fig.savefig(
            save_dir.joinpath(f"layer_{layer}_qkv_in_norm.png"),
            dpi=500,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    MODEL = "gpt2"
    # attn_qkv_in_norm(
    #   load_model(MODEL),
    #   save_dir=f"img/attn_weight_norm/{MODEL}"
    # )
    # attn_weight_norm(
    #   load_model(MODEL),
    #   save_dir=f"img/attn_weight_norm/{MODEL}"
    # )
    # attn_qk_norm(
    #   load_model(MODEL),
    #   save_dir=f"img/attn_weight_norm/{MODEL}"
    # )
    weight_norm(
        load_model(MODEL),
        save_dir=f"img/weight_norm/{MODEL}",
    )
