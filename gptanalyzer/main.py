import hydra
from omegaconf import DictConfig

from gptanalyzer.modules import end_hydra_run, init_hydra_run, init_logging

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


@hydra.main(version_base="1.2", config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """_summary_

    Parameters
    ----------
    cfg : DictConfig
        _description_
    """

    output_dir = init_hydra_run(cfg)
    # CODE HERE

    end_hydra_run(cfg, output_dir)


if __name__ == "__main__":
    main()
