from typing import Tuple
from deepxube.base.environment import Environment
from deepxube.base.heuristic import NNetPar
import math
import re


def get_environment(env_name: str) -> Tuple[Environment, NNetPar]:
    env_name = env_name.lower()
    puzzle_n_regex = re.search(r"puzzle(\d+)", env_name)

    if (env_name == "cube3") or (env_name == "cube3_fixed"):
        from deepxube.environments.cube3 import Cube3, Cube3NNetParV
        return Cube3(env_name == "cube3_fixed"), Cube3NNetParV()
    elif puzzle_n_regex is not None:
        from deepxube.environments.n_puzzle import NPuzzle
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        return NPuzzle(env_name, puzzle_dim)
    elif env_name == 'sokoban':
        from deepxube.environments.sokoban import Sokoban
        return Sokoban(env_name)
    else:
        raise ValueError('No known environment %s' % env_name)
