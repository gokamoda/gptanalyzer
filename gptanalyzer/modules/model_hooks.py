from typing import Literal

from torch import nn
from torch.utils.hooks import RemovableHandle
from torchtyping import TensorType

from gptanalyzer.models.gpt2 import MyGPT2LMHeadModel
from gptanalyzer.models.gptneox import MyGPTNeoXForCausalLM

from .my_dataclasses import (BatchHookResultForAttention,
                             BatchHookResultForLayer, BatchHookResultForLN,
                             BatchHookResultForMLP, BatchHookResultForModel)
from .my_torchtyping import BATCH, HEAD, KEY, QUERY

from gptanalyzer.modules.mylogger import init_logging
import sys

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)



class Hook:
    """Base class for hooks."""

    hook: RemovableHandle

    def __init__(
        self, module: nn.Module, mode: Literal["attn", "layer", "mlp", "ln"]
    ) -> None:
        self.hook = module.register_forward_hook(
            self.hook_fn, with_kwargs=True
        )
        self.mode = mode
        self.result = None

    def hook_fn(self, module, args, kwargs, output) -> None:
        """Hook function to catch attention weights."""
        del module, args, output
        self.result = {
            "attn": BatchHookResultForAttention,
            "layer": BatchHookResultForLayer,
            "ln": BatchHookResultForLN,
            "mlp": BatchHookResultForMLP,
        }[self.mode](**kwargs)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()


class HookForAttnTokenIntervention:
    """Hook for attention ablation."""

    def __init__(
        self,
        module: nn.Dropout,
        mode: Literal["fix", "ablate"],
        ablation_token_positions: list[tuple[int, int, int, int]] = None,
        fix_attention_weights: TensorType[BATCH, HEAD, QUERY, KEY] = None,
    ):
        """Hook for attention ablation.

        Parameters
        ----------
        module : nn.Dropout
            e.g., transformer.h[layer_idx].attn.attn_dropout
        ablation_position : tuple[int, int, int, int]
            (batch_idx, head_idx, query_idx, key_idx)
        """
        if mode == "ablate":
            self.hook = module.register_forward_pre_hook(self.ablate_hook_fn)
            self.ablation_token_positions = ablation_token_positions

        elif mode == "fix":
            self.hook = module.register_forward_pre_hook(self.fix_hook_fn)
            self.fix_attention_weights = fix_attention_weights

    def ablate_hook_fn(self, module, dropout_input):
        """Ablate attention weights."""
        del module
        dropout_input_list = list(dropout_input)
        # input[0].shape =
        # (batch_size, n_head, query_len, key_len) = (1, 12, 13, 13)
        for prompt, head, query, key in self.ablation_token_positions:
            dropout_input_list[0][prompt, head, query, key] = 0

        return tuple(dropout_input_list)

    def fix_hook_fn(self, module, dropout_input):
        """Fix attention weights."""
        del module
        dropout_input_list = list(dropout_input)
        dropout_input_list[0] = self.fix_attention_weights.to(
            dropout_input_list[0].device
        )

        return tuple(dropout_input_list)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()


class HookForAttnDimentionAblation:
    """Hook for attention ablation by dimention."""

    def __init__(
        self,
        module: nn.Dropout,
        ablation_dimention_positions: list[tuple[int, int, int]],
    ):
        """Hook for attention ablation.

        Parameters
        ----------
        module : nn.Dropout
            e.g., transformer.h[layer_idx].attn.value_dropout
        ablation_position : tuple[int, int, int]
            (batch_idx, head_idx, dim_idx)
        """
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.ablation_dimention_positions = ablation_dimention_positions

    def hook_fn(self, module, dropout_input):
        """Ablate attention weights."""
        del module
        dropout_input_list = list(dropout_input)
        for prompt, head, dim in self.ablation_dimention_positions:
            dropout_input_list[0][prompt, head, :, dim] = 0
            # dropout_input_list[0][0, head, :, dim] = dropout_input_list[0][
            #     0, head, :, dim
            # ].mean()

        return tuple(dropout_input_list)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()


class HookForMLPDimentionsAblation:
    """Hook for attention ablation by dimention."""

    def __init__(
        self,
        module: nn.Dropout,
        ablation_dimention_positions: list[tuple[int, int]],
    ):
        """Hook for attention ablation.

        Parameters
        ----------
        module : nn.Dropout
            e.g., transformer.h[layer_idx].mlp.dropout
        ablation_position : tuple[int, int]
            (batch_idx, dim_idx)
        """
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.ablation_dimention_positions = ablation_dimention_positions

    def hook_fn(self, module, dropout_input):
        """Ablate attention weights."""
        del module
        dropout_input_list = list(dropout_input)
        for prompt, dim in self.ablation_dimention_positions:
            dropout_input_list[0][prompt, :, dim] = 0

        return tuple(dropout_input_list)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()


def set_observation_hooks(
    model: MyGPT2LMHeadModel | MyGPTNeoXForCausalLM,
    class_field_names: dict[str, str],
    layer_hook: bool,
    attention_hook: bool,
    ln_hook: bool,
    mlp_hook: bool,
) -> dict[str, dict[str, Hook]]:
    """Set hooks for observation.

    Parameters
    ----------
    model : MyGPT2LMHeadModel
    layer_hook : bool
    attention_hook : bool

    Returns
    -------
    dict[int, dict[str, Hook]]
    """
    n_layer = getattr(model.config, class_field_names["n_layer"])
    observation_hooks = {str(layer_idx): {} for layer_idx in range(n_layer)}

    if layer_hook:
        for layer_idx in range(n_layer):
            observation_hooks[str(layer_idx)]["layer"] = Hook(
                getattr(
                    getattr(model, class_field_names["model_class_name"]),
                    class_field_names["layer_class_name"],
                )[layer_idx].for_hook,
                mode="layer",
            )

    if attention_hook:
        for layer_idx in range(n_layer):
            observation_hooks[str(layer_idx)]["attn"] = Hook(
                getattr(
                    getattr(
                        getattr(model, class_field_names["model_class_name"]),
                        class_field_names["layer_class_name"],
                    )[layer_idx],
                    class_field_names["attention_class_name"],
                ).for_hook,
                mode="attn",
            )

    if mlp_hook:
        for layer_idx in range(n_layer):
            observation_hooks[str(layer_idx)]["mlp"] = Hook(
                getattr(
                    getattr(model, class_field_names["model_class_name"]),
                    class_field_names["layer_class_name"],
                )[layer_idx].mlp.for_hook,
                mode="mlp",
            )

    if ln_hook:
        for layer_idx in range(n_layer):
            for layer_norm_class_name in class_field_names[
                "layer_norm_class_names"
            ]:
                observation_hooks[str(layer_idx)][layer_norm_class_name] = (
                    Hook(
                        getattr(
                            getattr(
                                getattr(
                                    model,
                                    class_field_names["model_class_name"],
                                ),
                                class_field_names["layer_class_name"],
                            )[layer_idx],
                            layer_norm_class_name,
                        ).for_hook,
                        mode="ln",
                    )
                )

        observation_hooks[class_field_names["ln_f"]] = Hook(
            getattr(
                getattr(model, class_field_names["model_class_name"]),
                class_field_names["ln_f"],
            ).for_hook,
            mode="ln",
        )

    return observation_hooks


def get_observation_hooks_results(
    observation_hooks: dict[str, dict[str, Hook]],
    class_field_names: dict[str, str],
    n_layer: int,
) -> BatchHookResultForModel:
    """_summary_

    Parameters
    ----------
    observation_hooks : dict[int, dict[str, Hook]]
        hooks set by set_observation_hooks
    layer_hook : bool
        _description_
    attention_hook : bool
        _description_
    n_layer : int
        _description_

    Returns
    -------
    list[BatchHookResultForLayer]
        _description_
    """
    if "layer" in observation_hooks["0"].keys():
        hook_results = [
            observation_hooks[str(layer_idx)]["layer"].result
            for layer_idx in range(n_layer)
        ]
    else:
        hook_results = [BatchHookResultForLayer() for _ in range(n_layer)]

    for layer_idx in range(n_layer):
        if "attn" in observation_hooks[str(layer_idx)].keys():
            hook_results[layer_idx].attn = observation_hooks[str(layer_idx)][
                "attn"
            ].result
        if "mlp" in observation_hooks[str(layer_idx)].keys():
            hook_results[layer_idx].mlp = observation_hooks[str(layer_idx)][
                "mlp"
            ].result
        if "ln_1" in observation_hooks[str(layer_idx)].keys():
            hook_results[layer_idx].ln_1 = observation_hooks[str(layer_idx)][
                "ln_1"
            ].result
        if "ln_2" in observation_hooks[str(layer_idx)].keys():
            hook_results[layer_idx].ln_2 = observation_hooks[str(layer_idx)][
                "ln_2"
            ].result

    if "ln_f" in observation_hooks.keys():
        return BatchHookResultForModel(
            h=hook_results, ln_f=observation_hooks["ln_f"].result
        )
    return BatchHookResultForModel(h=hook_results)


def remove_observation_hooks(
    observation_hooks: dict[int, dict[str, Hook]]
) -> None:
    """Remove hooks set by set_observation_hooks.

    Parameters
    ----------
    observation_hooks : dict[int, dict[str, Hook]]
        hooks set by set_observation_hooks
    """
    for layer_idx in observation_hooks.keys():
        if isinstance(observation_hooks[layer_idx], dict):
            for key in observation_hooks[layer_idx].keys():
                observation_hooks[layer_idx][key].remove()
        else:
            observation_hooks[layer_idx].remove()


def set_attn_ablation_hooks(
    attention_token_ablation: dict[int, list[tuple[int, int, int]]],
    attention_dimention_ablation: dict[int, list[tuple[int, int]]],
    model: MyGPT2LMHeadModel,
) -> list[HookForAttnTokenIntervention]:
    """Set hooks for attention ablation.

    Parameters
    ----------
    attention_ablation : dict[int, list[tuple[int, int, int]]]
    model : MyGPT2LMHeadModel

    Returns
    -------
    list[HookForAttnAblation]
    """
    ablation_hooks = []

    if attention_token_ablation is not None:
        for layer_idx, positions in attention_token_ablation.items():
            ablation_hooks.append(
                HookForAttnTokenIntervention(
                    module=model.transformer.h[layer_idx].attn.attn_dropout,
                    mode="ablate",
                    ablation_token_positions=positions,
                )
            )

    if attention_dimention_ablation is not None:
        for layer_idx, positions in attention_dimention_ablation.items():
            ablation_hooks.append(
                HookForAttnDimentionAblation(
                    module=model.transformer.h[
                        layer_idx
                    ].attn.attn_value_dropout,
                    ablation_dimention_positions=positions,
                )
            )

    return ablation_hooks


def set_mlp_ablation_hooks(
    mlp_dimention_ablation: dict[int, list[tuple[int, int]]],
    model: MyGPT2LMHeadModel,
) -> list[HookForMLPDimentionsAblation]:
    """Set hooks for attention ablation.

    Parameters
    ----------
    attention_ablation : dict[int, list[tuple[int, int, int]]]
    model : MyGPT2LMHeadModel

    Returns
    -------
    list[HookForAttnAblation]
    """
    ablation_hooks = []

    if mlp_dimention_ablation is not None:
        for layer_idx, positions in mlp_dimention_ablation.items():
            ablation_hooks.append(
                HookForMLPDimentionsAblation(
                    module=model.transformer.h[layer_idx].mlp.dropout,
                    ablation_dimention_positions=positions,
                )
            )

    return ablation_hooks


def fix_attn_weights(
    data: BatchHookResultForModel,
    model: MyGPT2LMHeadModel,
):
    """Add hooks to control attention weights."""
    fix_hooks: list[HookForAttnTokenIntervention] = [
        HookForAttnTokenIntervention(
            module=model.transformer.h[layer_idx].attn.attn_dropout,
            mode="fix",
            fix_attention_weights=layer.attn.attn_weights,
        )
        for layer_idx, layer in enumerate(data.h)
    ]
    return fix_hooks


def remove_attn_intervention_hooks(
    ablation_hooks: list[HookForAttnTokenIntervention],
) -> None:
    """Remove hooks set by set_attn_ablation_hooks
    and check if ablation is successful.

    Parameters
    ----------
    ablation_hooks : list[HookForAttnAblation]
        Ablation hooks set by set_attn_ablation_hooks
    assertion : bool, optional
        Whether to check if ablation is successful, by default True
    attentions : TensorType[LAYER, HEAD, SEQUENCE, SEQUENCE], optional
        Attentions from model.generate(). Necessary for assertion,
        by default None
    attention_ablation : dict[int, list[tuple[int, int, int]]], optional
        Attention ablation positions. Necessary for assertion, by default None

    Raises
    ------
    AssertionError
        When attentions are not provided for assertion.
    AssertionError
        When attention_ablation is not provided for assertion.
    AssertionError
        When ablation failed.
    """

    for hook in ablation_hooks:
        hook.remove()
