import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchtyping import TensorType
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import (GPT2MLP, GPT2Attention,
                                                    GPT2Block, GPT2Model)

from gptanalyzer.modules.my_torchtyping import (BATCH, HEAD, HIDDEN_DIM,
                                                SEQUENCE)
from gptanalyzer.modules.mylogger import init_logging

from .hook import ForwardHook

torch.set_printoptions(sci_mode=False)

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


class MyGPT2Attention(GPT2Attention):
    """GPT2Attention with hooks pre-computed wvo and bvo."""

    wvo = None
    bvo = None
    attn_w = None
    attn_b = None
    wqkh = None
    bqwkh = None

    def __init__(self, config, layer_idx):
        del layer_idx
        super().__init__(config)
        self.for_hook = ForwardHook()
        self.attn_value_dropout = nn.Dropout(0)

    def _my_attn(
        self, hidden_states, value, attention_mask=None, head_mask=None
    ) -> Tuple[
        TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM],
        TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE],
    ]:
        """_summary_

        Parameters
        ----------
        query : _type_
            _description_
        key : _type_
            _description_
        value : _type_
            _description_
        attention_mask : _type_, optional
            _description_, by default None
        head_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        Tuple[
            weighted_value:
                TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM],
                attn_output = values_by_head_and_seq.sum(dim=-2)
            attn_weights: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE],
        ]
            _description_
        """
        _, s, _ = hidden_states.size()

        compare_score = torch.einsum(
            "bshd,bdt->bhst",
            torch.einsum(
                "bsd,hdi->bshi",
                hidden_states,
                self.wqkh.to(hidden_states.device),
            ),
            hidden_states.transpose(-1, -2),
        )

        gate_score = torch.einsum(
            "hd,bdt->bht",
            self.bqwkh.to(hidden_states.device),
            hidden_states.transpose(-1, -2),
        )

        attn_weights = compare_score + gate_score.unsqueeze(-2)

        if self.scale_attn_weights:
            compare_score = compare_score / torch.full(
                [],
                self.head_dim**0.5,
                dtype=compare_score.dtype,
                device=compare_score.device,
            )

            attn_weights = attn_weights / torch.full(
                [],
                # value.size(-1)
                self.head_dim**0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            raise NotImplementedError

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = s, s
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error:
            #   `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise
            #   `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full(
                [],
                mask_value,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            compare_score = torch.where(
                causal_mask, compare_score.to(compare_score.dtype), mask_value
            )

            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        compare_score = nn.functional.softmax(compare_score, dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # logger.info(gate_score_dim_ablate[0][0])

        # Downcast (if necessary) back to V's dtype (if in mixed-precision)
        #  -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value = self.attn_value_dropout(value)
        weighted_value = torch.einsum("bhij,bhjd->bhijd", attn_weights, value)

        # attn_output = torch.matmul(attn_weights, value)

        return weighted_value, attn_weights

    def collapse_ln(
        self,
        ln_weight: TensorType[HIDDEN_DIM],
        ln_bias: TensorType[HIDDEN_DIM],
    ):
        """Collapse weights and biases of ln_1.

        Parameters
        ----------
        ln_weight : TensorType[HIDDEN_DIM]
        ln_bias : TensorType[HIDDEN_DIM]
        """

        centering = (
            torch.diag(torch.ones(ln_weight.shape[0])) - 1 / ln_weight.shape[0]
        )
        centering = centering.to(ln_weight.device)

        self.c_attn.bias = nn.Parameter(
            ln_bias @ self.c_attn.weight + self.c_attn.bias
        )
        self.c_attn.weight = nn.Parameter(
            centering @ torch.diag(ln_weight) @ self.c_attn.weight
        )

    def compute_wvo(self):
        """Pre-compute wvo and bvo."""
        wv = self.c_attn.weight[:, -self.embed_dim :]
        bv = self.c_attn.bias[-self.embed_dim :]
        wo = self.c_proj.weight
        bo = self.c_proj.bias

        wvh = wv.T.view(
            self.num_heads, self.head_dim, self.embed_dim
        ).transpose(-1, -2)
        woh = wo.view(self.num_heads, self.head_dim, self.embed_dim)

        wvo = wvh @ woh  # shape = (num_heads, embed_dim, embed_dim)
        bvo = bv @ wo + bo

        self.wvo = wvo
        self.bvo = bvo

    def compute_wqk(self):
        """Pre-compute wqk."""
        wq = self.c_attn.weight[:, : self.embed_dim]
        wk = self.c_attn.weight[:, self.embed_dim : self.embed_dim * 2]
        bq = self.c_attn.bias[: self.embed_dim]

        wqh = wq.T.view(
            self.num_heads, self.head_dim, self.embed_dim
        ).transpose(-1, -2)
        wkh = wk.T.view(self.num_heads, self.head_dim, self.embed_dim)
        self.wqkh = wqh @ wkh  # (h, d, d)

        bqh = bq.view(self.num_heads, self.head_dim)
        self.bqwkh = torch.einsum("he,hed->hd", bqh, wkh)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            raise NotImplementedError

        value = torch.einsum(
            "bsd,hdi->bhsi", hidden_states, self.wvo.to(hidden_states.device)
        )

        if layer_past is not None:
            raise NotImplementedError

        if use_cache is True:
            raise NotImplementedError

        present = None

        if self.reorder_and_upcast_attn:
            raise NotImplementedError

        (
            weighted_value,
            attn_weights,
        ) = self._my_attn(hidden_states, value, attention_mask, head_mask)

        # + TATSURO
        _ = self.for_hook(
            attn_weights=attn_weights.detach().to("cpu"),
            weighted_value=weighted_value.detach().to("cpu"),
            # key=key.detach().to("cpu"),
            # query=query.detach().to("cpu"),
            # original_value=original_value.detach().to("cpu"),
        )

        attn_output: TensorType[BATCH, SEQUENCE, HIDDEN_DIM] = (
            weighted_value.sum(dim=-2).permute(0, 2, 1, 3)
        ).sum(dim=2)
        attn_output = attn_output + self.bvo.to(hidden_states.device)
        # attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class MyGPT2MLP(GPT2MLP):
    """GPT2MLP with hooks."""

    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size=intermediate_size, config=config)
        self.for_hook = ForwardHook()

    def collapse_ln(
        self,
        ln_weight: TensorType[HIDDEN_DIM],
        ln_bias: TensorType[HIDDEN_DIM],
    ):
        """Collapse weights and biases of ln_2.

        Parameters
        ----------
        ln_weight : TensorType[HIDDEN_DIM]
        ln_bias : TensorType[HIDDEN_DIM]
        """

        centering = (
            torch.diag(torch.ones(ln_weight.shape[0])) - 1 / ln_weight.shape[0]
        )
        centering = centering.to(ln_weight.device)

        self.c_fc.bias = nn.Parameter(
            ln_bias @ self.c_fc.weight + self.c_fc.bias
        )
        self.c_fc.weight = nn.Parameter(
            centering @ torch.diag(ln_weight) @ self.c_fc.weight
        )

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        activation = self.act(hidden_states)
        hidden_states = self.c_proj(activation)
        hidden_states = self.dropout(hidden_states)
        self.for_hook(
            activation=activation.detach().to("cpu")
        )  # remove batch dim
        return hidden_states


class MyLayerNorm(nn.LayerNorm):
    """LayerNorm with hooks."""

    def __init__(self, config, collapsed=False):
        super().__init__(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mean = None
        self.var = None
        self.for_hook = ForwardHook()
        self.collapsed = collapsed

    # pylint: disable=arguments-renamed
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """LN with intervention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            _description_

        Returns
        -------
        torch.FloatTensor
            _description_
        """
        if self.mean is not None:
            mean = self.mean
        else:
            mean = hidden_states.mean(dim=-1, keepdim=True)

        if self.var is not None:
            var = self.var
        else:
            var = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        norm = hidden_states / torch.sqrt(var + 1e-5)
        self.for_hook(
            mean=mean.detach().to("cpu"),
            var=var.detach().to("cpu"),
        )
        self.mean = None
        self.var = None
        # return norm
        if self.collapsed:
            return norm

        return norm * self.weight + self.bias


class MyGPT2Block(GPT2Block):
    """GPT2Block with hooks and pre-computed wvo and bvo."""

    def __init__(self, config, layer_idx):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = (
            config.n_inner if config.n_inner is not None else 4 * hidden_size
        )

        self.ln_1 = MyLayerNorm(config, collapsed=True)
        self.attn = MyGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = MyLayerNorm(config, collapsed=True)
        self.mlp = MyGPT2MLP(intermediate_size=inner_dim, config=config)
        self.for_hook = ForwardHook()

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual_before_attn = hidden_states

        # Layer normalization 1
        hidden_states = self.ln_1(hidden_states)

        # Attention
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual_before_attn

        if encoder_hidden_states is not None:
            raise NotImplementedError

        # Layer normalization 2
        residual_before_mlp = hidden_states
        hidden_states = self.ln_2(hidden_states)

        # MLP
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual_before_mlp + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        self.for_hook(
            residual_input=residual_before_attn.detach().to("cpu"),
            attn_output=attn_output.detach().to("cpu"),
            intermediate_residual=residual_before_mlp.detach().to("cpu"),
            mlp_output=feed_forward_hidden_states.detach().to("cpu"),
            residual_output=outputs[0].detach().to("cpu"),
        )
        return (
            outputs  # hidden_states, present, (attentions, cross_attentions)
        )


class MyGPT2Model(GPT2Model):
    """GPT2Model with hooks and pre-computed wvo and bvo."""

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList(
            [
                MyGPT2Block(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = MyLayerNorm(config, collapsed=True)
        self.post_init()


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT2LMHeadModel hooks and pre-computed wvo and bvo."""

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MyGPT2Model(config)

        # bias True to collapse from ln_f
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.post_init()

    def collapse_ln(
        self,
        ln_weight: TensorType[HIDDEN_DIM],
        ln_bias: TensorType[HIDDEN_DIM],
    ):
        """Collapse weights and biases of ln_f.

        Parameters
        ----------
        ln_weight : TensorType[HIDDEN_DIM]
        ln_bias : TensorType[HIDDEN_DIM]
        """
        centering = (
            torch.diag(torch.ones(ln_weight.shape[0])) - 1 / ln_weight.shape[0]
        )
        centering = centering.to(ln_weight.device)

        with torch.no_grad():
            self.lm_head.bias = nn.Parameter(ln_bias @ self.lm_head.weight.T)
            self.lm_head.weight = nn.Parameter(
                (centering @ torch.diag(ln_weight) @ self.lm_head.weight.T).T
            )


def load_model(model_name_or_path):
    """Load equivalent model for analysis."""
    model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path)
    for i in range(model.config.n_layer):
        model.transformer.h[i].attn.collapse_ln(
            ln_weight=model.transformer.h[i].ln_1.weight,
            ln_bias=model.transformer.h[i].ln_1.bias,
        )
        model.transformer.h[i].mlp.collapse_ln(
            ln_weight=model.transformer.h[i].ln_2.weight,
            ln_bias=model.transformer.h[i].ln_2.bias,
        )
        model.transformer.h[i].attn.compute_wvo()
        model.transformer.h[i].attn.compute_wqk()
    model.collapse_ln(
        model.transformer.ln_f.weight, model.transformer.ln_f.bias
    )
    logger.info("lm_head bias is True to collapse from ln_f.")
    return model
