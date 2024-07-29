import sys

import torch
from torchtyping import TensorType
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from gptanalyzer.models import MyGPT2LMHeadModel, init_logging
from gptanalyzer.modules.my_torchtyping import (HIDDEN_DIM, LAYER_PLUS_1,
                                                SEQUENCE, VOCAB)

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)

# def _logit_lens_


def _logit_lens_redefined_gpt2(
    model: MyGPT2LMHeadModel,
    hidden_states: TensorType[LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM],
    no_bias: bool = False,
    extended: bool = False,
):
    device = model.device
    lm_head = model.lm_head

    num_layers, _, _ = hidden_states.shape
    hidden_states = hidden_states.to(device)
    logits_by_layer = []

    layer_norm = model.transformer.ln_f
    if no_bias:
        lm_head.bias = torch.nn.Parameter(torch.zeros(lm_head.bias.shape))
    with torch.no_grad():
        for layer_idx, layer_hidden_state in enumerate(hidden_states):
            logits_by_token = []
            for _, token_hidden_state in enumerate(layer_hidden_state):
                token_hidden_state = token_hidden_state.to(device)
                if layer_idx in (num_layers - 1,) and not extended:
                    # last layer
                    # check transformers.models.gpt2.modeling_gpt2:914
                    y = lm_head(token_hidden_state)
                else:
                    y = lm_head(layer_norm(token_hidden_state))

                logits_by_token.append(y)
            logits_by_layer.append(torch.stack(logits_by_token).cpu())

    return torch.stack(logits_by_layer)


def logit_lens(
    model: MyGPT2LMHeadModel | AutoModelForCausalLM,
    hidden_states: TensorType[LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM],
    no_bias: bool = False,
    extended: bool = False,
) -> TensorType[LAYER_PLUS_1, SEQUENCE, VOCAB]:
    """logit lens (nostalgebraist)

    Parameters
    ----------
    model : MyGPT2LMHeadModel
    hidden_states : TensorType[LAYERS_PLUS_1, SEQUENCE, HIDDEN_DIM]

    Returns
    -------
    TensorType[LAYERS_PLUS_1, SEQUENCE, VOCAB]
    """

    device = model.device
    lm_head = model.lm_head

    num_layers, _, _ = hidden_states.shape
    hidden_states = hidden_states.to(device)
    logits_by_layer = []

    if isinstance(model, MyGPT2LMHeadModel):
        return _logit_lens_redefined_gpt2(
            model=model,
            hidden_states=hidden_states,
            no_bias=no_bias,
            extended=extended,
        )

    if model.config.architectures[0] == "OPTForCausalLM":
        layer_norm = model.model.decoder.final_layer_norm
    elif isinstance(model, LlamaForCausalLM):
        layer_norm = model.model.norm
    else:
        layer_norm = model.transformer.ln_f

    if no_bias:
        layer_norm.bias = torch.nn.Parameter(
            torch.zeros(layer_norm.bias.shape)
        )

    with torch.no_grad():
        for layer_idx, layer_hidden_state in enumerate(hidden_states):
            logits_by_token = []
            for _, token_hidden_state in enumerate(layer_hidden_state):
                token_hidden_state = token_hidden_state.to(device)
                if layer_idx in (num_layers - 1,):
                    # last layer
                    # check transformers.models.gpt2.modeling_gpt2:914
                    y = lm_head(token_hidden_state)
                else:
                    if no_bias:
                        y = lm_head(layer_norm(token_hidden_state))
                    else:
                        y = lm_head(layer_norm(token_hidden_state))

                logits_by_token.append(y)
            logits_by_layer.append(torch.stack(logits_by_token).cpu())

    return torch.stack(logits_by_layer)
