import sys

from gptanalyzer.modules.mylogger import init_logging

from .gpt2 import MyGPT2LMHeadModel
from .gpt2 import load_model as load_gpt2_model
from .gptneox import load_model as load_gptneox_model

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def load_model(
    model_name_or_path: str,
) -> MyGPT2LMHeadModel:
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
        return load_gpt2_model(model_name_or_path)
    if model_name_or_path.startswith("EleutherAI/pythia"):
        return load_gptneox_model(model_name_or_path)

    raise NotImplementedError
