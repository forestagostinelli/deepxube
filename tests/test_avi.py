from deepxube.environments.environment_abstract import Environment
from deepxube.environments.env_utils import get_environment
from deepxube.training.avi import Status


env_name: str = "cube3"


def test_status():
    env: Environment = get_environment(env_name)
    step_max: int = 30
    num_per_step: int = 2
    status: Status = Status(env, step_max, num_per_step)
    num_states_gen: int = (step_max + 1) * num_per_step
    assert status.per_solved_best > 0
    assert len(status.state_t_steps_l) == num_states_gen
    assert len(status.states_start_t) == num_states_gen
    assert len(status.goals_t) == num_states_gen
    assert status.itr == 0
    assert status.update_num == 0
