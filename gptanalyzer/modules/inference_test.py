# pylint: disable=duplicate-code

import sys

import torch
from transformers import GPT2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding

from gptanalyzer.analyzers import logit_lens
from gptanalyzer.models import load_model
from gptanalyzer.models.models_test import call

from .inference import attn_weight_repeat, generate, ln_data_repeat
from .my_dataclasses import (BatchHookResultForAttention, BatchHookResultForLN,
                             BatchHookResultForMLP, BatchHookResultForModel,
                             Instance)
from .mylogger import init_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


def test_generate():
    """Check if __call__ result can be reproduced with model.generate()."""

    prompt = "Tokyo is the capital of"
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    result = Instance(
        prompt=prompt,
        inputs=tokenizer(prompt, return_tensors="pt"),
    )

    call_logits = call(model, inputs).logits

    result.hf_generation = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        layer_hook=True,
        output_hidden_states=True
    )

    # check if model.generate() returns the same info as model.__call__
    assert torch.allclose(
        logit_lens(
            model=model,
        class_field_names=class_field_names,
            hidden_states=result.hf_generation.hidden_states[0],
        )[-1][-1],
        call_logits[-1][-1],
    )


def test_observation_hooks():
    """observation hooks test"""

    prompt = "Tokyo is the capital of"
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    result = Instance(
        prompt=prompt,
        inputs=tokenizer(prompt, return_tensors="pt"),
    )

    result.hf_generation = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        mlp_hook=True,
        layer_hook=True,
        ln_hook=True,
        output_hidden_states=True
    )

    assert isinstance(result.hf_generation.hidden_states, torch.Tensor)
    assert isinstance(
        result.hf_generation.hook_results, BatchHookResultForModel
    )
    assert isinstance(result.hf_generation.hook_results.h, list)
    assert isinstance(
        result.hf_generation.hook_results.ln_f, BatchHookResultForLN
    )
    assert len(result.hf_generation.hook_results.h) == model.config.n_layer
    assert isinstance(
        result.hf_generation.hook_results.h[0].attn,
        BatchHookResultForAttention,
    )
    assert isinstance(
        result.hf_generation.hook_results.h[0].mlp, BatchHookResultForMLP
    )


def test_attn_token_ablation():
    """attention token ablation test"""

    prompt = "Tokyo is the capital of"
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    attn_token_ablation = {0: [(0, 1, 3, 1)], 1: [(0, 1, 4, 2)]}
    attn_token_ablation_neg = {0: [(0, 1, 3, 0)], 1: [(0, 1, 4, 1)]}

    result = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        attention_token_ablation=attn_token_ablation,
    )

    for layer_idx, positions in attn_token_ablation.items():
        for prompt, head, query, key in positions:
            assert (
                result.hook_results.h[layer_idx].attn.attn_weights[
                    prompt, head, query, key
                ]
                == 0
            )

    for layer_idx, positions in attn_token_ablation_neg.items():
        for prompt, head, query, key in positions:
            assert (
                result.hook_results.h[layer_idx].attn.attn_weights[
                    prompt, head, query, key
                ]
                != 0
            )


def test_attn_dim_ablation():
    """attention dimention ablation test"""

    prompt = ["Tokyo is the capital of", "Tokyo is the capital of"]
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    attn_dim_ablation = {0: [(0, 1, 3)], 1: [(0, 1, 4)]}
    attn_dim_ablation_neg = {0: [(0, 1, 4)], 1: [(1, 1, 4)]}

    result = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        attention_dimention_ablation=attn_dim_ablation,
    )

    for layer_idx, positions in attn_dim_ablation.items():
        for prompt, head, dim in positions:
            assert torch.all(
                result.hook_results.h[layer_idx]
                .attn.weighted_value[prompt, head, :, :, dim]
                .flatten()
                == 0
            )

    for layer_idx, positions in attn_dim_ablation_neg.items():
        for prompt, head, dim in positions:
            assert not torch.all(
                result.hook_results.h[layer_idx]
                .attn.weighted_value[prompt, head, :, :, dim]
                .flatten()
                == 0
            )


def test_mlp_dim_ablation():
    """mlp dimention ablation test"""
    prompt = ["Tokyo is the capital of", "Tokyo is the capital of"]
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    mlp_dim_ablation = {0: [(0, 1)], 1: [(0, 3)]}
    mlp_dim_ablation_neg = {0: [(0, 2)], 1: [(1, 3)]}

    result = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        layer_hook=True,
        mlp_dimention_ablation=mlp_dim_ablation,
    )

    for layer_idx, positions in mlp_dim_ablation.items():
        for prompt, dim in positions:
            assert torch.all(
                result.hook_results.h[layer_idx]
                .mlp_output[prompt, :, dim]
                .flatten()
                == 0
            )

    for layer_idx, positions in mlp_dim_ablation_neg.items():
        for prompt, dim in positions:
            assert not torch.all(
                result.hook_results.h[layer_idx]
                .mlp_output[prompt, :, dim]
                .flatten()
                == 0
            )


def test_ln_fix():
    """LayerNorm fix test"""
    prompts = ["Tokyo is the capital of", "Tokyo is is is is"]
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompts, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    original = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        ln_hook=True,
    ).unbatch()

    ln_fix_data = ln_data_repeat(data=original[0].hook_results, n=2)
    inputs: BatchEncoding = tokenizer(prompts, return_tensors="pt")

    ln_fix_result = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        ln_hook=True,
        ln_fix=ln_fix_data,
    ).unbatch()

    # check if the ln fix is working
    assert torch.allclose(
        original[0].hook_results.ln_f.mean,
        ln_fix_result[1].hook_results.ln_f.mean,
    )
    assert torch.allclose(
        original[0].hook_results.ln_f.var,
        ln_fix_result[1].hook_results.ln_f.var,
    )
    assert not torch.allclose(
        original[1].hook_results.h[0].ln_1.mean,
        ln_fix_result[1].hook_results.h[0].ln_1.mean,
    )


def test_attn_fix():
    """Attention fix test"""
    prompts = ["Tokyo is the capital of", "Tokyo is is is is"]
    model_name_or_path = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompts, return_tensors="pt")

    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    original = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
    ).unbatch()

    attn_fix_data = attn_weight_repeat(data=original[0].hook_results, n=2)
    inputs: BatchEncoding = tokenizer(prompts, return_tensors="pt")

    attn_fix_result = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        attn_fix=attn_fix_data,
    ).unbatch()

    # check if the ln fix is working
    assert torch.allclose(
        original[0].hook_results.h[0].attn.attn_weights,
        attn_fix_result[1].hook_results.h[0].attn.attn_weights,
    )
    assert not torch.allclose(
        original[1].hook_results.h[0].attn.attn_weights,
        attn_fix_result[1].hook_results.h[0].attn.attn_weights,
    )
