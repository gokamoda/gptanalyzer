import sys
from typing import Tuple

from gptanalyzer.modules.mylogger import init_logging

from .gpt2 import MyGPT2LMHeadModel
from .gpt2 import load_model as load_gpt2_model
from .gptneox import MyGPTNeoXForCausalLM
from .gptneox import load_model as load_gptneox_model

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def load_model(
    model_name_or_path: str,
) -> Tuple[MyGPT2LMHeadModel | MyGPTNeoXForCausalLM, dict[str, str]]:
    """Load model with hooks and pre-computed wvo and bvo.

    Parameters
    ----------
    model_name_or_path : str
        Pretrained model name or path from huggingface.

    Returns
    -------
    MyGPT2LMHeadModel
        Model with hooks and pre-computed wvo and bvo.

    Raises
    ------
    NotImplementedError
        If the model is not supported.
    """
    if model_name_or_path.startswith("gpt2"):
        class_field_names = {
            "n_layer": "n_layer",
            "model_class_name": "transformer",
            "layer_class_name": "h",
            "attention_class_name": "attn",
            "layer_norm_class_names": ["ln_1", "ln_2"],
            "ln_f": "ln_f",
            "lm_head": "lm_head",
        }
        return load_gpt2_model(model_name_or_path), class_field_names
    if model_name_or_path.startswith("EleutherAI/pythia"):
        class_field_names = {
            "n_layer": "num_hidden_layers",
            "model_class_name": "gpt_neox",
            "layer_class_name": "layers",
            "attention_class_name": "attention",
            "layer_norm_class_names": [
                "input_layernorm",
                "post_attention_layernorm",
            ],
            "ln_f": "final_layer_norm",
            "lm_head": "embed_out",
        }
        return load_gptneox_model(model_name_or_path), class_field_names

    raise NotImplementedError
