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

from gptanalyzer.modules.my_torchtyping import (BATCH, HEAD, HEAD_DIM,
                                                HIDDEN_DIM, SEQUENCE)
from gptanalyzer.modules.mylogger import init_logging
from gptanalyzer.nn_utils import ForwardHook, MyLayerNorm

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

    wvo = None
    bvo = None

    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.for_hook = ForwardHook()

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

        # self.value.weight = nn.Parameter(qkvw[2])
        # self.value.bias = nn.Parameter(qkvb[2])

        wvh: TensorType[HEAD, HIDDEN_DIM, HEAD_DIM] = (
            qkvw[2]
            .view(self.num_attention_heads, self.head_size, self.hidden_size)
            .transpose(-1, -2)
        )
        woh: TensorType[HEAD, HEAD_DIM, HIDDEN_DIM] = self.dense.weight.T.view(
            self.num_attention_heads, self.head_size, self.hidden_size
        )
        self.wvo: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM] = wvh @ woh
        self.bvo: TensorType[HIDDEN_DIM] = (
            qkvb[2] @ self.dense.weight.T + self.dense.bias
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = (
            query.size()
        )
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        query = query.view(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.view(
            batch_size * num_attention_heads, key_length, attn_head_size
        )
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error:
        # `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise
        # `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device
        )
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        # attn_output = torch.matmul(attn_weights, value)
        attn_output = torch.einsum("bhij,bhjd->bhijd", attn_weights, value)
        return attn_output, attn_weights

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
        # attn_output = self._merge_heads(
        #     attn_output, self.num_attention_heads, self.head_size
        # )
        # attn_output = self.dense(attn_output)
        self.for_hook(
            attn_weights=attn_weights.detach().to("cpu"),
        )
        attn_output: TensorType[
            BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM
        ] = (
            attn_output.sum(dim=-2)  # sum by key position
            .permute(0, 2, 1, 3)
            .sum(dim=2)  # sum by head
        )

        attn_output = attn_output + self.bvo

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

        # Compute QKV values
        new_shape = hidden_states.size()[:-1] + (
            self.num_attention_heads,
            self.head_size,
        )

        query: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = (
            self.query(hidden_states).view(new_shape).permute(0, 2, 1, 3)
        )
        key: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = (
            self.key(hidden_states).view(new_shape).permute(0, 2, 1, 3)
        )
        value: TensorType[BATCH, HEAD, SEQUENCE, HIDDEN_DIM] = torch.einsum(
            "bsd,hdi->bhsi", hidden_states, self.wvo
        )

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
            raise NotImplementedError()

        if use_cache:
            raise NotImplementedError("use_cache is not implemented yet.")

        return query, key, value, None


class MyGPTNeoXMLP(GPTNeoXMLP):
    """Custom MLP layer for GPTNeoX."""

    def __init__(self, config):
        super().__init__(config)
        self.for_hook = ForwardHook()

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        self.for_hook(activation=hidden_states.detach().to("cpu"))
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class MyGPTNeoXLayer(GPTNeoXLayer):
    """GPTNeoXLayer with custom attention and mlp layers."""

    def __init__(self, config):
        super().__init__(config)
        self.input_layernorm = MyLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = MyLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = MyGPTNeoXAttention(config)
        self.mlp = MyGPTNeoXMLP(config)
        self.for_hook = ForwardHook()

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual_input = hidden_states
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
            self.for_hook(
                residual_input=residual_input.detach().to("cpu"),
                attn_output=attn_output.detach().to("cpu"),
                mlp_output=mlp_output.detach().to("cpu"),
                residual_output=hidden_states.detach().to("cpu"),
            )
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            intermediate_residual = attn_output + hidden_states
            mlp_output = self.mlp(
                self.post_attention_layernorm(intermediate_residual)
            )
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output
            self.for_hook(
                residual_input=residual_input.detach().to("cpu"),
                attn_output=attn_output.detach().to("cpu"),
                intermediate_residual=intermediate_residual.detach().to("cpu"),
                mlp_output=mlp_output.detach().to("cpu"),
                residual_output=hidden_states.detach().to("cpu"),
            )

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[
                1:
            ]  # hidden_states, (attn_weights)

        return outputs


class MyGPTNeoXModel(GPTNeoXModel):
    """GPTNeoXModel with custom attention and mlp layers."""

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MyGPTNeoXLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = MyLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
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
