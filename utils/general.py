import sys

from typing import List, TypeVar

ListElement = TypeVar('ListElement')


def flatten_list_of_lists(
        list_of_lists: List[List[ListElement]]
) -> List[ListElement]:
    return [e for sub_list in list_of_lists for e in sub_list]


def out_of_menu_exit(text: str) -> None:
    print(f"The {text} is out menu...")
    sys.exit()
