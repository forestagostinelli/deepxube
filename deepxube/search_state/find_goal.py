from typing import Optional, List, Tuple, Any
from deepxube.search_state.astar import AStar, Node, get_path
from deepxube.environments.environment_abstract import EnvGrndAtoms, State, HeurFnNNet, Goal
from deepxube.logic.program import Clause, Model
from deepxube.specifications.asp import ASPSpec
from deepxube.utils.timing_utils import Times
import time


def state_model_to_goal_node(env: EnvGrndAtoms, state_start: State, model: Optional[Model], batch_size: int,
                             weight: float, max_search_itrs: int, heur_fn: HeurFnNNet) -> Optional[Node]:
    # Check for None
    if model is None:
        return None

    # Initialize
    goal: Goal = env.model_to_goal([model])[0]
    astar = AStar(env)
    astar.add_instances([state_start], [goal], [weight], heur_fn)

    # Search
    search_itr: int = 0
    while (not min(x.finished for x in astar.instances)) and (search_itr < max_search_itrs):
        search_itr += 1
        astar.step(heur_fn, batch_size, verbose=False)

    return astar.instances[0].goal_node


def get_next_model(asp: ASPSpec, spec_clauses: List[Clause], env: EnvGrndAtoms, models_banned: List[Model],
                   min_model_atoms: Optional[int] = None) -> Optional[Model]:
    if min_model_atoms is not None:
        breakpoint()
    models: List[Model] = asp.get_models(spec_clauses, env.on_model, minimal=True, num_models=1,
                                         models_banned=models_banned)
    if len(models) == 0:
        return None
    else:
        return models[0]


def find_goal(env: EnvGrndAtoms, state_start: State, spec_clauses: List[Clause], heur_fn: HeurFnNNet,
              model_itrs_max: int, batch_size: int, weight: float, max_search_itrs: int,
              times: Optional[Times] = None,
              verbose: bool = False) -> Tuple[bool, List[State], List[Any], float, Times]:
    """

    :param env: EnvGrndAtoms environment
    :param state_start: Starting state
    :param spec_clauses: Clauses for specification. All must have goal in the head.
    :param heur_fn: Heuristic function
    :param model_itrs_max: Maximum number of randomly sampled models to try
    :param batch_size: Batch size of search
    :param weight: Weight on path cost for weighted search. Must be between 0 and 1.
    :param max_search_itrs: Maximum number of iterations when searching from start state to goal model
    :param times: Times
    :param verbose: Prints if true
    :return: boolean that is true if a goal is found, list of states along path, list of actions along path, path cost,
    times
    """
    # Init
    if times is None:
        times = Times()
    models_banned: List[Model] = []

    # Initialize ASP
    start_time = time.time()
    asp: ASPSpec = ASPSpec(env.get_ground_atoms(), env.get_bk())
    times.record_time("ASP init", time.time() - start_time)
    for model_itr in range(model_itrs_max):
        # Get initial model
        start_time = time.time()
        model: Optional[Model] = get_next_model(asp, spec_clauses, env, models_banned)
        if verbose:
            print("Initial model: ", model)
        times.record_time("Model init", time.time() - start_time)

        # Find goal node
        goal_node: Optional[Node] = state_model_to_goal_node(env, state_start, model, batch_size, weight,
                                                             max_search_itrs, heur_fn)
        while goal_node is not None:
            # determine if goal state has been found
            state_terminal = goal_node.state
            model_terminal: Model = env.state_to_model([state_terminal])[0]
            is_model: bool = asp.check_model(spec_clauses, model_terminal)

            if is_model:
                # return path to goal
                if verbose:
                    print("Found a goal state")
                path_states, path_actions, path_cost = get_path(goal_node)
                return True, path_states, path_actions, path_cost, times

            # Get next model
            start_time = time.time()
            model: Optional[Model] = get_next_model(asp, spec_clauses, env, models_banned,
                                                    min_model_atoms=len(model) + 1)
            if verbose:
                print("Next model: ", model)
            times.record_time("Model superset", time.time() - start_time)

            # Find goal node
            goal_node: Optional[Node] = state_model_to_goal_node(env, state_start, model, batch_size, weight,
                                                                 max_search_itrs, heur_fn)
        if model is not None:
            models_banned.append(model)

        if verbose:
            print("Goal node is None")

    return False, [], [], -1.0, times
