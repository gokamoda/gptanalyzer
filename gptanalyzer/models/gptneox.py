import sys
from typing import Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType
from transformers.models.gpt_neox.modeling_gpt_neox import (GPTNeoXAttention,
                                                            GPTNeoXForCausalLM,
                                                            GPTNeoXLayer,
                                                            GPTNeoXMLP,
                                                            GPTNeoXModel)

from gptanalyzer.models.hook import ForwardHook
from gptanalyzer.modules.my_torchtyping import BATCH, HIDDEN_DIM, SEQUENCE
from gptanalyzer.modules.mylogger import init_logging

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and
            key tensors. For example, this can be used to pass offsetted
            position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which
            to unsqueeze cos[position_ids] and sin[position_ids] so that they
            can be properly broadcasted to the dimensions of q and k. For
            example, note that cos[position_ids] and sin[position_ids] have
            the shape [batch_size, seq_len, head_dim]. Then, if q and k have
            the shape [batch_size, heads, seq_len, head_dim], then setting
            unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids]
            broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set
            unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated
        using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MyGPTNeoXAttention(GPTNeoXAttention):
    """Custom attention layer for GPTNeoX."""

    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def split_qkv_weight(self):
        """Prepare QKV weights for attention computation."""
        qkvw = (
            self.query_key_value.weight.view(
                self.num_attention_heads, 3, self.head_size, self.hidden_size
            )
            .permute(1, 0, 2, 3)
            .reshape(3, self.hidden_size, self.hidden_size)
        )
        qkvb = (
            self.query_key_value.bias.view(
                self.num_attention_heads, 3, self.head_size
            )
            .permute(1, 0, 2)
            .reshape(3, self.hidden_size)
        )

        self.query.weight = nn.Parameter(qkvw[0])
        self.query.bias = nn.Parameter(qkvb[0])

        self.key.weight = nn.Parameter(qkvw[1])
        self.key.bias = nn.Parameter(qkvb[1])

        self.value.weight = nn.Parameter(qkvw[2])
        self.value.bias = nn.Parameter(qkvb[2])

    def forward(
        self,
        hidden_states: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
        )

        # Compute attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        # Reshape outputs
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        new_shape = hidden_states.size()[:-1] + (
            self.num_attention_heads,
            self.head_size,
        )
        query = self.query(hidden_states).view(new_shape).permute(0, 2, 1, 3)
        key = self.key(hidden_states).view(new_shape).permute(0, 2, 1, 3)
        value = self.value(hidden_states).view(new_shape).permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids
        )
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        return query, key, value, present


class MyGPTNeoXMLP(GPTNeoXMLP):
    """Custom MLP layer for GPTNeoX."""

    def __init__(self, config):
        super().__init__(config)
        self.for_hook = ForwardHook()


class MyGPTNeoXLayer(GPTNeoXLayer):
    """GPTNeoXLayer with custom attention and mlp layers."""

    def __init__(self, config):
        super().__init__(config)
        self.attention = MyGPTNeoXAttention(config)
        self.mlp = MyGPTNeoXMLP(config)


class MyGPTNeoXModel(GPTNeoXModel):
    """GPTNeoXModel with custom attention and mlp layers."""

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MyGPTNeoXLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_init()


class MyGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """GPTNeoXForCausalLM with custom attention and mlp layers."""

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = MyGPTNeoXModel(config)


def load_model(model_name_or_path):
    """Load equivalent model for analysis."""
    model = MyGPTNeoXForCausalLM.from_pretrained(model_name_or_path)
    for i in range(model.config.num_hidden_layers):
        model.gpt_neox.layers[i].attention.split_qkv_weight()
    # for i in range(model.config.n_layer):
    #     model.transformer.h[i].attn.collapse_ln(
    #         ln_weight=model.transformer.h[i].ln_1.weight,
    #         ln_bias=model.transformer.h[i].ln_1.bias,
    #     )
    #     model.transformer.h[i].mlp.collapse_ln(
    #         ln_weight=model.transformer.h[i].ln_2.weight,
    #         ln_bias=model.transformer.h[i].ln_2.bias,
    #     )
    #     model.transformer.h[i].attn.compute_wvo()
    #     model.transformer.h[i].attn.compute_wqk()
    # model.collapse_ln(
    #     model.transformer.ln_f.weight, model.transformer.ln_f.bias
    # )
    # logger.info("lm_head bias is True to collapse from ln_f.")
    return model
