import sys

import torch
from transformers.tokenization_utils_base import BatchEncoding

from gptanalyzer.models.gpt2 import MyGPT2LMHeadModel
from gptanalyzer.models.gptneox import MyGPTNeoXForCausalLM

from .model_hooks import (Hook, fix_attn_weights,
                          get_observation_hooks_results,
                          remove_attn_intervention_hooks,
                          remove_observation_hooks, set_attn_ablation_hooks,
                          set_mlp_ablation_hooks, set_observation_hooks)
from .my_dataclasses import (BatchHookResultForAttention,
                             BatchHookResultForLayer, BatchHookResultForLN,
                             BatchHookResultForModel,
                             BatchHuggingfaceGenerationPlus,
                             HookResultForModel)
from .mylogger import init_logging

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def ln_data_repeat(
    data: HookResultForModel, n: int
) -> BatchHookResultForModel:
    """Repeat the layer norm data n times for batch processing.

    Parameters
    ----------
    data : HookResultForModel
        _description_
    n : int
        _description_

    Returns
    -------
    BatchHookResultForModel
        _description_
    """
    return BatchHookResultForModel(
        ln_f=BatchHookResultForLN(
            mean=data.ln_f.mean.unsqueeze(0).repeat(n, 1, 1),
            var=data.ln_f.var.unsqueeze(0).repeat(n, 1, 1),
        ),
        h=[
            BatchHookResultForLayer(
                ln_1=BatchHookResultForLN(
                    mean=h.ln_1.mean.unsqueeze(0).repeat(n, 1, 1),
                    var=h.ln_1.var.unsqueeze(0).repeat(n, 1, 1),
                ),
                ln_2=BatchHookResultForLN(
                    mean=h.ln_2.mean.unsqueeze(0).repeat(n, 1, 1),
                    var=h.ln_2.var.unsqueeze(0).repeat(n, 1, 1),
                ),
            )
            for h in data.h
        ],
    )


def attn_weight_repeat(
    data: HookResultForModel, n: int
) -> BatchHookResultForModel:
    """Repeat the attention weights data n times for batch processing.

    Parameters
    ----------
    data : HookResultForModel
        _description_
    n : int
        _description_

    Returns
    -------
    BatchHookResultForModel
        _description_
    """
    return BatchHookResultForModel(
        h=[
            BatchHookResultForLayer(
                attn=BatchHookResultForAttention(
                    attn_weights=h.attn.attn_weights.unsqueeze(0).repeat(
                        n, 1, 1, 1
                    )
                ),
            )
            for h in data.h
        ],
    )


def generate(
    model: MyGPT2LMHeadModel | MyGPTNeoXForCausalLM,
    class_field_names: dict[str, str],
    inputs: BatchEncoding,
    pad_token_id: int,
    output_attentions: bool = False,
    output_logits: bool = False,
    output_hidden_states: bool = False,
    attention_hook: bool = False,
    mlp_hook: bool = False,
    ln_hook: bool = False,
    layer_hook: bool = False,
    ln_fix: BatchHookResultForModel = None,
    attn_fix: BatchHookResultForModel = None,
    attention_token_ablation: dict[int, list[tuple[int, int, int]]] = None,
    attention_dimention_ablation: dict[int, list[tuple[int, int]]] = None,
    mlp_dimention_ablation: dict[int, list[tuple[int, int]]] = None,
) -> BatchHuggingfaceGenerationPlus:
    """Run inference on the model.

    Parameters
    ----------
    model : MyGPT2LMHeadModel
        Model to use.
    input_ids : BatchEncoding
        Inputs with input_ids and attention_mask.
    pad_token_id : int
        Token id for padding.
    attention_hook : bool
        Whether to use attention hook.
    mlp_hook : bool
        Whether to use mlp hook.
    layer_hook : bool
        Whether to use layer hook.
    attention_token_ablation : dict[int, list[tuple[int, int, int]]]
        Ablation token positions for attention.
        Example: {0: [(1, 3, 1), (1, 4, 2)]}\r
        Will ablate the attention at
        (layer 0, head 1, query 3, key 1) and
        (layer 0, head 1, query 4, key 2).
    attention_dimention_ablation : dict[int, list[tuple[int, int, int]]]
        Ablation dimention positions in tuple(batch, head, dim)for attention.
        Example: {0: [(0, 1, 3), (2, 1, 4)]}\r
        Will ablate the attention at
        (layer 0, prompt 0, head 1, query 3) and
        (layer 0, prompt 2, head 1, query 4).

    Returns
    -------
    _type_
        _description_
    """

    # if inputs["input_ids"].shape[0] != 1:
    #     raise NotImplementedError("Batch processing not implemented.")

    model.eval()

    observation_hooks: dict[str, dict[str, Hook]] | None = (
        set_observation_hooks(
            model=model,
            class_field_names=class_field_names,
            layer_hook=layer_hook,
            attention_hook=attention_hook,
            ln_hook=ln_hook,
            mlp_hook=mlp_hook,
        )
        if attention_hook or mlp_hook or layer_hook or ln_hook
        else None
    )

    ablation_hooks: list[Hook] | None = None
    if (
        attention_token_ablation is not None
        or attention_dimention_ablation is not None
        or mlp_dimention_ablation is not None
    ):
        ablation_hooks = []
        if (
            attention_dimention_ablation is not None
            or attention_token_ablation is not None
        ):
            ablation_hooks += set_attn_ablation_hooks(
                model=model,
                attention_token_ablation=attention_token_ablation,
                attention_dimention_ablation=attention_dimention_ablation,
            )
        if mlp_dimention_ablation is not None:
            ablation_hooks += set_mlp_ablation_hooks(
                model=model,
                mlp_dimention_ablation=mlp_dimention_ablation,
            )

    if ln_fix is not None:
        for layer_idx, hook_result in enumerate(ln_fix.h):
            model.transformer.h[layer_idx].ln_1.mean = (
                hook_result.ln_1.mean.to(model.device)
            )
            model.transformer.h[layer_idx].ln_1.var = hook_result.ln_1.var.to(
                model.device
            )
            model.transformer.h[layer_idx].ln_2.mean = (
                hook_result.ln_2.mean.to(model.device)
            )
            model.transformer.h[layer_idx].ln_2.var = hook_result.ln_2.var.to(
                model.device
            )
        model.transformer.ln_f.mean = ln_fix.ln_f.mean.to(model.device)
        model.transformer.ln_f.var = ln_fix.ln_f.var.to(model.device)

    if attn_fix is not None:
        if ablation_hooks is None:
            ablation_hooks = []
        ablation_hooks += fix_attn_weights(model=model, data=attn_fix)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=1,
            pad_token_id=pad_token_id,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_logits=output_logits,
        )

    observation_hook_results = (
        None
        if observation_hooks is None
        else get_observation_hooks_results(
            observation_hooks=observation_hooks,
            class_field_names=class_field_names,
            n_layer=getattr(model.config, class_field_names["n_layer"]),
        )
    )

    if observation_hooks is not None:
        remove_observation_hooks(observation_hooks)

    if ablation_hooks is not None:
        remove_attn_intervention_hooks(
            ablation_hooks=ablation_hooks,
        )

    return BatchHuggingfaceGenerationPlus(
        hidden_states=torch.stack(output.hidden_states[0], dim=1).to("cpu") if output_hidden_states else None,
        attentions=torch.stack(output.attentions[0], dim=1).to("cpu") if output_attentions else None,
        logits=output.logits[0].to("cpu") if output_logits else None,
        generated_tokens=output.sequences.to("cpu"),
        hook_results=observation_hook_results,
    )
