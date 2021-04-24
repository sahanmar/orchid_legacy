import sys

from typing import List, Dict

from utils.util_types import Morphology


def out_of_menu_exit(text: str) -> None:
    print(f"The {text} is out menu...")
    sys.exit()
