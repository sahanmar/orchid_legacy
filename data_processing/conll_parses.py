import argparse
import sys

from pathlib import Path
from typing import Tuple, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to conll input file")
    return parser.parse_args()


def path_check(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    return None


if __name__ == "__main__":
    args = parse_args()
    path = path_check(Path(args.input))
    if not path:
        print(f"The path '{args.input}' is not a file")
        sys.exit()
