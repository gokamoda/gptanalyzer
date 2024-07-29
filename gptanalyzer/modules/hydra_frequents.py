import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .mylogger import init_logging

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


def init_hydra_run(cfg: DictConfig) -> Path:
    """_summary_

    Parameters
    ----------
    cfg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")
    logger.info(OmegaConf.to_yaml(cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir_str = hydra_cfg.runtime.output_dir
    output_dir = Path(output_dir_str)
    logger.info(output_dir)

    save_latest_dir = Path(cfg.save_latest_dir)
    if save_latest_dir.exists():
        shutil.rmtree(save_latest_dir)
    save_latest_dir.mkdir(parents=True)

    return output_dir


def end_hydra_run(cfg: DictConfig, output_dir: Path) -> None:
    """Copy logs and outputs to the output directory

    Parameters
    ----------
    cfg : DictConfig
        cfg from hydra
    output_dir : Path
        hydra output directory
        Should be returned by init_hydra_run
    """
    shutil.copy(LOG_PATH, f"{cfg.save_latest_dir}/main.log")
    shutil.copytree(cfg.save_latest_dir, output_dir.joinpath("latest"))
    logger.info("log file copied to logs/%s/latest/main.log", output_dir)
