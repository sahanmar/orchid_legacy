import logging
import warnings
from typing import Dict, Optional

VERBOSITY_MAPPING = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}


def get_logging_level(verbosity: int = 2, verbosity_mapping: Optional[Dict[int, int]] = None) -> int:
    if verbosity_mapping is None:
        global VERBOSITY_MAPPING
        verbosity_mapping = VERBOSITY_MAPPING
    _lvl = verbosity_mapping.get(int(verbosity))
    if _lvl is None:
        warnings.warn(f'Invalid passed verbosity level, defaulting to DEBUG: {verbosity}')
        _lvl = logging.DEBUG
    return _lvl


def get_stream_logger(
        name: str,
        verbosity: int = 2,
        log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(get_logging_level(verbosity=verbosity))

    # Stream formatter & handler
    fh = logging.Formatter(fmt=log_format)
    sh = logging.StreamHandler()
    sh.setFormatter(fh)

    logger.addHandler(sh)
    return logger


if __name__ == '__main__':
    pass
