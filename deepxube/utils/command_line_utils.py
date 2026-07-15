from typing import Tuple, Optional, List
import sys
import shlex


def print_command() -> None:
    print(" ".join(shlex.quote(arg) for arg in sys.argv))


def get_name_args(name_args: str) -> Tuple[str, Optional[str]]:
    name_args_split: List[str] = name_args.split(".", 1)
    name: str = name_args_split[0]
    args: Optional[str]
    if len(name_args_split) == 1:
        args = None
    else:
        assert len(name_args_split) == 2
        args = name_args_split[1]
    return name, args
