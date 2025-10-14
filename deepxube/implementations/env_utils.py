from typing import Tuple
from deepxube.base.env import Env
from deepxube.base.heuristic import NNetPar


def get_environment(env_name: str) -> Tuple[Env, NNetPar]:
    env_name = env_name.lower()

    if (env_name == "cube3") or (env_name == "cube3_fixed"):
        from deepxube.implementations.cube3 import Cube3, Cube3NNetParV
        return Cube3(env_name == "cube3_fixed"), Cube3NNetParV()
    else:
        raise ValueError('No known environment %s' % env_name)
