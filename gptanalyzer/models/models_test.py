# pylint: disable=duplicate-code

import sys

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

from gptanalyzer.models import load_model
from gptanalyzer.modules.mylogger import init_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


def call(model, inputs):
    """Call model with inputs."""
    model.eval()

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
        )
    return output


def test_gpt2_model():
    """Check if redefined model has the same output as the original model."""

    prompt = "Tokyo is the capital of"
    model_name_or_path = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    # ORIGINAL
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    original = call(model, inputs)

    # REDEFINED
    model, _ = load_model(model_name_or_path)
    model = model.to(device)
    redefined = call(model, inputs)

    logger.info(original.logits)
    logger.info(redefined.logits[0])
    assert torch.allclose(original.logits, redefined.logits[0])


def test_gptneox_model():
    """Check if redefined model has the same output as the original model."""

    prompt = "Tokyo is the capital of"
    model_name_or_path = "EleutherAI/pythia-14m"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    # ORIGINAL
    model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path).to(device)
    original = call(model, inputs)

    # REDEFINED
    model, _ = load_model(model_name_or_path)
    model = model.to(device)
    redefined = call(model, inputs)

    # set tensor precision to compare
    torch.set_printoptions(precision=10)
    logger.info(original.logits[0, 0])
    logger.info(redefined.logits[0][0])

    # rtol default was 1e-5
    # this error is predicted to be the cause of joining attention.value and
    # attention.dense linear transformations.
    assert torch.allclose(original.logits, redefined.logits[0], rtol=2e-5)
