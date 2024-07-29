import logging
import sys
from pathlib import Path


def init_logging(
    logger_name: str, log_path: str = "logs/info.log", clear=False
) -> logging.Logger:
    """_summary_

    Parameters
    ----------
    logger_name : str
        _description_
    log_path : str, optional
        _description_, by default "logs/info.log"

    Returns
    -------
    logging.Logger
        _description_
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    path = Path(log_path)

    dir_path = path.parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    if clear:
        with open(log_path, "w", encoding="utf-8"):
            pass

    if logger.handlers == []:
        dir_path.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s/%(levelname)s/%(name)s/%(funcName)s():%(lineno)s\n"
            "%(message)s\n"
        )

        # file handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.propagate = False

    return logger
