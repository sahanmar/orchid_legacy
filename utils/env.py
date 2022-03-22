import os
from pathlib import Path
from typing import Dict, Optional, Any

from dotenv import dotenv_values


def resolve_env() -> str:
    return os.environ.get('ORCHID_ENV', 'dev')


def get_env_configuration(
        path_shared: Path,
        path_secret: Optional[Path] = None,
        **kwargs
) -> Dict[str, Any]:
    if path_secret is not None:
        _secret_vals = dotenv_values(dotenv_path=path_secret, **kwargs)
    else:
        _secret_vals = {}
    return {
        **dotenv_values(dotenv_path=path_shared, **kwargs),
        **_secret_vals,
        **os.environ
    }
