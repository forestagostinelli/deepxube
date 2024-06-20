from deepxube.environments.environment_abstract import Environment
import math
import re


def get_environment(env_name: str) -> Environment:
    env_name = env_name.lower()
    puzzle_n_regex = re.search(r"puzzle(\d+)", env_name)

    if re.match('^cube3$', env_name):
        from deepxube.environments.cube3 import Cube3
        return Cube3(env_name)
    elif puzzle_n_regex is not None:
        from deepxube.environments.n_puzzle import NPuzzle
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        return NPuzzle(env_name, puzzle_dim)
    elif env_name == 'sokoban':
        from deepxube.environments.sokoban import Sokoban
        return Sokoban(env_name)
    else:
        raise ValueError('No known environment %s' % env_name)
