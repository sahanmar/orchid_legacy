import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Any

from dotenv import dotenv_values

__ENV_NAME_VAR = 'ORCHID_ENV'


def resolve_env(default_env: str = 'dev') -> str:
    env_name = os.environ.get(__ENV_NAME_VAR)
    if env_name is None:
        warnings.warn(
            f'Environment type is not specified by \"{__ENV_NAME_VAR}\";'
            f' default value \"{default_env}\" will be used'
        )
        env_name = default_env
    return env_name


def get_env_variables(
        path_shared: Path,
        path_secret: Optional[Path] = None,
        **kwargs
) -> Dict[str, Any]:
    if path_secret is not None:
        _secret_vals = dotenv_values(dotenv_path=path_secret, **kwargs)
        overwrite_with_secret = bool(int(_secret_vals.get('OVERWRITE_VARS')))
    else:
        _secret_vals = {}
        overwrite_with_secret = False
    return {
        **os.environ,
        **dotenv_values(dotenv_path=path_shared, **kwargs),
        **_secret_vals
    } if overwrite_with_secret else {
        **os.environ,
        **_secret_vals,
        **dotenv_values(dotenv_path=path_shared, **kwargs)
    }
