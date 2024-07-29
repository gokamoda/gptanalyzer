from dataclasses import dataclass
from pathlib import Path

import torch
from torchtyping import TensorType
from transformers.tokenization_utils_base import BatchEncoding

from .my_torchtyping import (BATCH, HEAD, HEAD_DIM, HIDDEN_DIM, LAYER,
                             LAYER_PLUS_1, MLP_ACTIVATION, SEQUENCE, VOCAB)


@dataclass
class HookResultForAttention:
    """Dataclass for a hook result of attention layer."""

    attn_weights: TensorType[HEAD, SEQUENCE, SEQUENCE] = None
    weighted_value: TensorType[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM] = None
    key: TensorType[HEAD, SEQUENCE, HEAD_DIM] = None
    query: TensorType[HEAD, SEQUENCE, HEAD_DIM] = None
    original_value: TensorType[HEAD, SEQUENCE, HEAD_DIM] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HookResultForAttention:\n"
        msg += (
            f"\tattn_weights: {self.attn_weights.shape}\n"
            if self.attn_weights is not None
            else ""
        )
        msg += (
            f"\tweighted_value: {self.weighted_value.shape}\n"
            if self.weighted_value is not None
            else ""
        )
        msg += f"\tkey: {self.key.shape}\n" if self.key is not None else ""
        msg += (
            f"\tquery: {self.query.shape}\n" if self.query is not None else ""
        )
        return msg


@dataclass
class BatchHookResultForAttention:
    """Dataclass for a hook result of attention layer."""

    attn_weights: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE] = None
    weighted_value: TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM] = (
        None
    )
    key: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = None
    query: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = None
    original_value: TensorType[BATCH, HEAD, SEQUENCE, HEAD_DIM] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHookResultForAttention:\n"
        msg += (
            f"\tattn_weights: {self.attn_weights.shape}\n"
            if self.attn_weights is not None
            else ""
        )
        msg += (
            f"\tweighted_value: {self.weighted_value.shape}\n"
            if self.weighted_value is not None
            else ""
        )
        msg += f"\tkey: {self.key.shape}\n" if self.key is not None else ""
        msg += (
            f"\tquery: {self.query.shape}\n" if self.query is not None else ""
        )
        return msg

    def unbatch(self, idx: int) -> HookResultForAttention:
        """Unbatch the batched tensors."""
        return HookResultForAttention(
            attn_weights=(
                self.attn_weights[idx]
                if self.attn_weights is not None
                else None
            ),
            weighted_value=(
                self.weighted_value[idx]
                if self.weighted_value is not None
                else None
            ),
            key=self.key[idx] if self.key is not None else None,
            query=self.query[idx] if self.query is not None else None,
        )

    def save(self, name, save_root: Path, dtype=torch.float16):
        """Save the instance to a file."""
        if self.attn_weights is not None:
            attn_weights_dir = save_root.joinpath("attn_weights")
            attn_weights_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.attn_weights.to(dtype),
                attn_weights_dir.joinpath(f"{name}.pt"),
            )

        if self.weighted_value is not None:
            weighted_value_dir = save_root.joinpath("weighted_value")
            weighted_value_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.weighted_value.to(dtype),
                weighted_value_dir.joinpath(f"{name}.pt"),
            )

        if self.key is not None:
            key_dir = save_root.joinpath("key")
            key_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.key.to(dtype), key_dir.joinpath(f"{name}.pt"))

        if self.query is not None:
            query_dir = save_root.joinpath("query")
            query_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.query.to(dtype), query_dir.joinpath(f"{name}.pt"))


@dataclass
class HookResultForMLP:
    """Dataclass for a hook result of MLP layer."""

    activation: TensorType[SEQUENCE, MLP_ACTIVATION] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HookResultForMLP:\n"
        msg += (
            f"\tactivation: {self.activation.shape}\n"
            if self.activation is not None
            else ""
        )
        return msg


@dataclass
class BatchHookResultForMLP:
    """Dataclass for a hook result of MLP layer."""

    activation: TensorType[BATCH, SEQUENCE, MLP_ACTIVATION] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHookResultForMLP:\n"
        msg += (
            f"\tactivation: {self.activation.shape}\n"
            if self.activation is not None
            else ""
        )
        return msg

    def unbatch(self, idx: int) -> HookResultForMLP:
        """Unbatch the batched tensors."""
        return HookResultForMLP(
            activation=(
                self.activation[idx] if self.activation is not None else None
            )
        )

    def save(self, name, save_root: Path, dtype=torch.float16):
        """Save the instance to a file."""
        if self.activation is not None:
            activation_dir = save_root.joinpath("activation")
            activation_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.activation.to(dtype),
                activation_dir.joinpath(f"{name}.pt"),
            )


@dataclass
class HookResultForLN:
    """Dataclass for a hook result of LayerNorm layer."""

    mean: TensorType[SEQUENCE, 1] = None
    var: TensorType[SEQUENCE, 1] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HookResultForLN:\n"
        msg += f"\tmean: {self.mean.shape}\n" if self.mean is not None else ""
        msg += f"\tvar: {self.var.shape}\n" if self.var is not None else ""
        return msg


@dataclass
class BatchHookResultForLN:
    """Dataclass for a hook result of LayerNorm layer."""

    mean: TensorType[BATCH, SEQUENCE, 1] = None
    var: TensorType[BATCH, SEQUENCE, 1] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHookResultForLN:\n"
        msg += f"\tmean: {self.mean.shape}\n" if self.mean is not None else ""
        msg += f"\tvar: {self.var.shape}\n" if self.var is not None else ""
        return msg

    def unbatch(self, idx: int) -> HookResultForLN:
        """Unbatch the batched tensors."""
        return HookResultForLN(
            mean=self.mean[idx] if self.mean is not None else None,
            var=self.var[idx] if self.var is not None else None,
        )

    def save(
        self, name: str, save_root: Path, dtype: torch.dtype = torch.float16
    ):
        """Save hook results as .pt files.

        Parameters
        ----------
        name : str
        save_root : Path
            Directory to save the hook results.
        dtype : torch.dtype, optional
            torch.dtype to use for saving tensors, by default torch.float16
        """
        if self.mean is not None:
            mean_dir = save_root.joinpath("mean")
            mean_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.mean.to(dtype), mean_dir.joinpath(f"{name}.pt"))

        if self.var is not None:
            var_dir = save_root.joinpath("var")
            var_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.var.to(dtype), var_dir.joinpath(f"{name}.pt"))


@dataclass
class HookResultForLayer:
    """Dataclass for a hook result of a single layer."""

    attn: HookResultForAttention = None
    mlp: HookResultForMLP = None
    ln_1: HookResultForLN = None
    ln_2: HookResultForLN = None
    residual_input: TensorType[SEQUENCE, HIDDEN_DIM] = None
    attn_output: TensorType[SEQUENCE, HIDDEN_DIM] = None
    intermediate_residual: TensorType[SEQUENCE, HIDDEN_DIM] = None
    mlp_output: TensorType[SEQUENCE, HIDDEN_DIM] = None
    residual_output: TensorType[SEQUENCE, HIDDEN_DIM] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HookResultForLayer:\n"
        msg += "\tln_1: HookResultForLN\n" if self.ln_1 is not None else ""
        msg += (
            "\tattn: HookResultForAttention\n" if self.attn is not None else ""
        )
        msg += "\tln_2: HookResultForLN\n" if self.ln_2 is not None else ""
        msg += "\tmlp: HookResultForMLP\n" if self.mlp is not None else ""
        msg += (
            f"\tresidual_input: {self.residual_input.shape}\n"
            if self.residual_input is not None
            else ""
        )
        msg += (
            f"\tattn_output: {self.attn_output.shape}\n"
            if self.attn_output is not None
            else ""
        )
        msg += (
            f"\tintermediate_residual: {self.intermediate_residual.shape}\n"
            if self.intermediate_residual is not None
            else ""
        )
        msg += (
            f"\tmlp_output: {self.mlp_output.shape}\n"
            if self.mlp_output is not None
            else ""
        )
        msg += (
            f"\tresidual_output: {self.residual_output.shape}\n"
            if self.residual_output is not None
            else ""
        )
        return msg


@dataclass
class BatchHookResultForLayer:
    """Dataclass for a hook result of a single layer."""

    attn: BatchHookResultForAttention = None
    mlp: BatchHookResultForMLP = None
    ln_1: BatchHookResultForLN = None
    ln_2: BatchHookResultForLN = None
    residual_input: TensorType[SEQUENCE, HIDDEN_DIM] = None
    attn_output: TensorType[SEQUENCE, HIDDEN_DIM] = None
    intermediate_residual: TensorType[SEQUENCE, HIDDEN_DIM] = None
    mlp_output: TensorType[SEQUENCE, HIDDEN_DIM] = None
    residual_output: TensorType[SEQUENCE, HIDDEN_DIM] = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHookResultForLayer:\n"
        msg += (
            "\tln_1: BatchHookResultForLN\n" if self.ln_1 is not None else ""
        )
        msg += (
            "\tattn: BatchHookResultForAttention\n"
            if self.attn is not None
            else ""
        )
        msg += (
            "\tln_2: BatchHookResultForLN\n" if self.ln_2 is not None else ""
        )
        msg += "\tmlp: BatchHookResultForMLP\n" if self.mlp is not None else ""
        msg += (
            f"\tresidual_input: {self.residual_input.shape}\n"
            if self.residual_input is not None
            else ""
        )
        msg += (
            f"\tattn_output: {self.attn_output.shape}\n"
            if self.attn_output is not None
            else ""
        )
        msg += (
            f"\tintermediate_residual: {self.intermediate_residual.shape}\n"
            if self.intermediate_residual is not None
            else ""
        )
        msg += (
            f"\tmlp_output: {self.mlp_output.shape}\n"
            if self.mlp_output is not None
            else ""
        )
        msg += (
            f"\tresidual_output: {self.residual_output.shape}\n"
            if self.residual_output is not None
            else ""
        )
        return msg

    def unbatch(self, idx: int) -> HookResultForLayer:
        """Unbatch the batched tensors."""
        return HookResultForLayer(
            attn=self.attn.unbatch(idx) if self.attn is not None else None,
            mlp=self.mlp.unbatch(idx) if self.mlp is not None else None,
            ln_1=self.ln_1.unbatch(idx) if self.ln_1 is not None else None,
            ln_2=self.ln_2.unbatch(idx) if self.ln_2 is not None else None,
            residual_input=(
                self.residual_input[idx]
                if self.residual_input is not None
                else None
            ),
            attn_output=(
                self.attn_output[idx] if self.attn_output is not None else None
            ),
            intermediate_residual=(
                self.intermediate_residual[idx]
                if self.intermediate_residual is not None
                else None
            ),
            mlp_output=(
                self.mlp_output[idx] if self.mlp_output is not None else None
            ),
            residual_output=(
                self.residual_output[idx]
                if self.residual_output is not None
                else None
            ),
        )

    def save(
        self, name: str, save_root: Path, dtype: torch.dtype = torch.float16
    ):
        """Save hook results as .pt files.

        Parameters
        ----------
        name : str
        save_root : Path
            Directory to save the hook results.
        dtype : torch.dtype, optional
            torch.dtype to use for saving tensors, by default torch.float16
        """

        if self.attn is not None:
            attn_dir = save_root.joinpath("attn")
            attn_dir.mkdir(exist_ok=True, parents=True)
            self.attn.save(name, save_root=attn_dir, dtype=dtype)

        if self.mlp is not None:
            mlp_dir = save_root.joinpath("mlp")
            mlp_dir.mkdir(exist_ok=True, parents=True)
            self.mlp.save(name, save_root=mlp_dir, dtype=dtype)

        if self.ln_1 is not None:
            ln_1_dir = save_root.joinpath("ln_1")
            ln_1_dir.mkdir(exist_ok=True, parents=True)
            self.ln_1.save(name, save_root=ln_1_dir, dtype=dtype)

        if self.ln_2 is not None:
            ln_2_dir = save_root.joinpath("ln_2")
            ln_2_dir.mkdir(exist_ok=True, parents=True)
            self.ln_2.save(name, save_root=ln_2_dir, dtype=dtype)

        if self.residual_input is not None:
            residual_input_dir = save_root.joinpath("residual_input")
            residual_input_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.residual_input.to(dtype),
                residual_input_dir.joinpath(f"{name}.pt"),
            )

        if self.attn_output is not None:
            attn_output_dir = save_root.joinpath("attn_output")
            attn_output_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.attn_output.to(dtype),
                attn_output_dir.joinpath(f"{name}.pt"),
            )

        if self.intermediate_residual is not None:
            intermediate_residual_dir = save_root.joinpath(
                "intermediate_residual"
            )
            intermediate_residual_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.intermediate_residual.to(dtype),
                intermediate_residual_dir.joinpath(f"{name}.pt"),
            )

        if self.mlp_output is not None:
            mlp_output_dir = save_root.joinpath("mlp_output")
            mlp_output_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.mlp_output.to(dtype),
                mlp_output_dir.joinpath(f"{name}.pt"),
            )

        if self.residual_output is not None:
            residual_output_dir = save_root.joinpath("residual_output")
            residual_output_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.residual_output.to(dtype),
                residual_output_dir.joinpath(f"{name}.pt"),
            )


@dataclass
class HookResultForModel:
    """Dataclass for a hook result of a model."""

    h: list[HookResultForLayer] = None
    ln_f: HookResultForLN = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HookResultForModel:\n"
        msg += (
            "\th: list[HookResultForLayer]," f"len={len(self.h)}\n"
            if self.h is not None
            else ""
        )
        msg += "\tln_f: HookResultForLN\n" if self.ln_f is not None else ""
        return msg


@dataclass
class BatchHookResultForModel:
    """Dataclass for a hook result of a model."""

    h: list[BatchHookResultForLayer] = None
    ln_f: BatchHookResultForLN = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHookResultForModel:\n"
        msg += (
            "\th: list[BatchHookResultForLayer]," f"len={len(self.h)}\n"
            if self.h is not None
            else ""
        )
        msg += (
            "\tln_f: BatchHookResultForLN\n" if self.ln_f is not None else ""
        )
        return msg

    def unbatch(self, idx: int) -> HookResultForModel:
        """Unbatch the batched tensors."""
        return HookResultForModel(
            h=[layer.unbatch(idx) for layer in self.h],
            ln_f=self.ln_f.unbatch(idx) if self.ln_f is not None else None,
        )

    def save(self, name, save_root: Path, dtype=torch.float16):
        """Save the instance to a file."""
        if self.h is not None:
            for i, h in enumerate(self.h):
                save_layer_dir = save_root.joinpath(f"layer_{i}")
                h.save(
                    f"{name}_layer_{i}", save_root=save_layer_dir, dtype=dtype
                )

        if self.ln_f is not None:
            ln_f_dir = save_root.joinpath("ln_f")
            ln_f_dir.mkdir(exist_ok=True, parents=True)
            self.ln_f.save(name, save_root=ln_f_dir, dtype=dtype)


@dataclass
class HuggingfaceGenerationPlus:
    """Dataclass for a generation result of a single prompt."""

    hidden_states: TensorType[LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM]
    attentions: TensorType[LAYER, LAYER, HEAD, SEQUENCE, SEQUENCE]
    logits: TensorType[VOCAB]
    hook_results: HookResultForModel
    generated_tokens: TensorType[SEQUENCE]

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "HuggingfaceGeneration:\n"
        msg += (
            f"\thidden_states: {self.hidden_states.shape}\n"
            if self.hidden_states is not None
            else ""
        )
        msg += (
            f"\tattentions: {self.attentions.shape}\n"
            if self.attentions is not None
            else ""
        )
        msg += (
            f"\tlogits: {self.logits.shape}\n"
            if self.logits is not None
            else ""
        )
        msg += (
            "\thook_results: HookResultForModel\n"
            if self.hook_results is not None
            else ""
        )
        msg += (
            f"\tgenerated_tokens: {self.generated_tokens.shape}\n"
            if self.generated_tokens is not None
            else ""
        )
        return msg


@dataclass
class BatchHuggingfaceGenerationPlus:
    """Dataclass for a generation result of a single prompt."""

    hidden_states: TensorType[BATCH, LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM]
    attentions: TensorType[BATCH, LAYER, LAYER, HEAD, SEQUENCE, SEQUENCE]
    logits: TensorType[BATCH, VOCAB]
    hook_results: BatchHookResultForModel
    generated_tokens: TensorType[BATCH, SEQUENCE]

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "BatchHuggingfaceGeneration:\n"
        msg += (
            f"\thidden_states: {self.hidden_states.shape}\n"
            if self.hidden_states is not None
            else ""
        )
        msg += (
            f"\tattentions: {self.attentions.shape}\n"
            if self.attentions is not None
            else ""
        )
        msg += (
            f"\tlogits: {self.logits.shape}\n"
            if self.logits is not None
            else ""
        )
        msg += (
            "\thook_results: BatchHookResultForModel\n"
            if self.hook_results is not None
            else ""
        )
        msg += (
            f"\tgenerated_tokens: {self.generated_tokens.shape}\n"
            if self.generated_tokens is not None
            else ""
        )
        return msg

    def unbatch(self) -> list[HuggingfaceGenerationPlus]:
        """Unbatch the batched tensors."""
        batch_size = self.logits.shape[0]
        return [
            HuggingfaceGenerationPlus(
                hidden_states=self.hidden_states[idx],
                attentions=self.attentions[idx],
                logits=self.logits[idx],
                hook_results=self.hook_results.unbatch(idx),
                generated_tokens=self.generated_tokens[idx],
            )
            for idx in range(batch_size)
        ]

    def save(self, name: str, dtype=torch.float32) -> None:
        """Save the instance to a file."""
        save_root = Path("tensor_results")

        if self.hidden_states is not None:
            hidden_state_dir = save_root.joinpath("hidden_states")
            hidden_state_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.hidden_states.to(dtype),
                hidden_state_dir.joinpath(f"{name}.pt"),
            )

        if self.attentions is not None:
            attention_dir = save_root.joinpath("attentions")
            attention_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.attentions.to(dtype), attention_dir.joinpath(f"{name}.pt")
            )

        if self.logits is not None:
            logit_dir = save_root.joinpath("logits")
            logit_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.logits.to(dtype), logit_dir.joinpath(f"{name}.pt"))

        if self.hook_results is not None:
            hook_result_dir = save_root.joinpath("hook_results")
            hook_result_dir.mkdir(exist_ok=True, parents=True)
            self.hook_results.save(
                name, save_root=save_root.joinpath("hook_results"), dtype=dtype
            )

        if self.generated_tokens is not None:
            generated_token_dir = save_root.joinpath("generated_tokens")
            generated_token_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.generated_tokens.to(dtype),
                generated_token_dir.joinpath(f"{name}.pt"),
            )


@dataclass
class Instance:
    """Dataclass for a single prompt."""

    prompt: str | None = None
    tokenized_prompt: list[str] | None = None
    inputs: BatchEncoding | None = None
    hf_generation: HuggingfaceGenerationPlus | None = None
    generated_text: str | None = None
    logit_through_logit_lens: (
        TensorType[LAYER_PLUS_1, SEQUENCE, VOCAB] | None
    ) = None

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        msg = "Instance:\n"
        msg += f"\tprompt: {self.prompt}\n" if self.prompt is not None else ""
        msg += (
            f"\ttokenized_prompt: {self.tokenized_prompt}\n"
            if self.tokenized_prompt is not None
            else ""
        )
        msg += (
            "\tinputs: transformers.tokenization_utils_base.BatchEncoding\n"
            if self.inputs is not None
            else ""
        )
        msg += (
            "\thf_generation: modules.my_dataclasses.HuggingfaceGeneration\n"
            if self.hf_generation is not None
            else ""
        )
        msg += (
            f"\tgenerated_text: {self.generated_text}\n"
            if self.generated_text is not None
            else ""
        )
        msg += (
            f"\tlogit_through_logit_lens: "
            f"{self.logit_through_logit_lens.shape}\n"
            if self.logit_through_logit_lens is not None
            else ""
        )
        return msg
