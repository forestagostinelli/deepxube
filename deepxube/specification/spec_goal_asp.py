from typing import Optional, List, Tuple, Any
from deepxube.search.astar import AStar, Node, get_path
from deepxube.environments.environment_abstract import EnvGrndAtoms, State, Goal
from deepxube.nnet.nnet_utils import HeurFN_T
from deepxube.logic.logic_objects import Literal, Clause, Model, Atom
from deepxube.logic.logic_utils import atom_to_str
from deepxube.logic.asp import Solver, Spec
from deepxube.utils import viz_utils, misc_utils
from deepxube.utils.timing_utils import Times
import time


def get_bk(env: EnvGrndAtoms, bk_add_file_name: Optional[str]) -> List[str]:
    bk_init: List[str] = env.get_bk()
    bk_init.append("")

    if bk_add_file_name is not None:
        bk_add_file = open(bk_add_file_name, 'r')
        bk_init.extend(bk_add_file.read().split("\n"))
        bk_add_file.close()

    return bk_init


def search_for_goal(env: EnvGrndAtoms, state_start: State, models: List[Model], batch_size: int,
                    weight: float, max_search_itrs: int, heur_fn: HeurFN_T, search_verbose: bool,
                    viz_model: bool) -> List[Optional[Node]]:
    goals: List[Goal] = env.model_to_goal(models)

    # Check for None
    if len(models) == 0:
        return []

    # visualize
    if viz_model:
        print("Sampled goal visualization")
        viz_utils.visualize_examples(env, goals)

    # Initialize
    astar = AStar(env)
    astar.add_instances([state_start] * len(goals), goals, [weight] * len(goals), heur_fn)

    # Search
    search_itr: int = 0
    while (not min(x.finished for x in astar.instances)) and (search_itr < max_search_itrs):
        search_itr += 1
        astar.step(heur_fn, batch_size, verbose=search_verbose)

    return [x.goal_node for x in astar.instances]


def get_next_model(asp: Solver, spec_clauses: List[Clause], env: EnvGrndAtoms, models_banned: List[Model],
                   num_models: int, assumed_true: Optional[Model] = None,
                   num_atoms_gt: Optional[int] = None) -> List[Model]:
    if num_atoms_gt is not None:
        spec_clauses_new = []
        lit_count_gt: Literal = Literal("count_model_grnd_atoms_gt", (str(num_atoms_gt),), ("in",))
        for clause in spec_clauses:
            clause_new = Clause(clause.head, clause.body + (lit_count_gt,))
            spec_clauses_new.append(clause_new)
        spec_clauses = spec_clauses_new

    atoms_true: List[Atom] = []
    if assumed_true is not None:
        atoms_true = list(assumed_true)
    spec: Spec = Spec(goal_true=spec_clauses, atoms_true=atoms_true, models_banned=models_banned)
    models: List[Model] = asp.get_models(spec, env.on_model, num_models, True)

    return models


def path_to_spec_goal(env: EnvGrndAtoms, state_start: State, spec_clauses: List[Clause], heur_fn: HeurFN_T,
                      model_batch_size: int, search_batch_size: int, weight: float, max_search_itrs: int,
                      bk_add: Optional[str] = None, models_banned: Optional[List[Model]] = None,
                      times: Optional[Times] = None, spec_verbose: bool = False, search_verbose: bool = False,
                      viz_model: bool = False) -> Tuple[bool, List[State], List[Any], float, int, int, List[Model],
                                                        Times]:
    """

    :param env: EnvGrndAtoms environment
    :param state_start: Starting state
    :param spec_clauses: Clauses for specification. All must have goal in the head.
    :param heur_fn: Heuristic function
    :param model_batch_size: Maximum number of models sampled at once for parallelized search.
    :param search_batch_size: Batch size of search
    :param weight: Weight on path cost for weighted search. Must be between 0 and 1.
    :param max_search_itrs: Maximum number of iterations when searching from start state to goal model
    :param bk_add: A file for additional background information
    :param models_banned: Banned models
    :param times: Times
    :param spec_verbose: Verbose specification if true
    :param search_verbose: Verbose search if true
    :param viz_model: Set true to visualize each model before search
    :return: boolean that is true if a goal is found, list of states along path, list of actions along path, path cost,
    times
    """
    # Init
    if times is None:
        times = Times(time_names=["ASP init", "Model samp", "Search", "Check", "Model superset"])

    if models_banned is None:
        models_banned = []
    else:
        models_banned = models_banned.copy()

    # Initialize ASP
    start_time = time.time()
    model_fixed: Model = env.start_state_fixed([state_start])[0]
    bk: List[str] = get_bk(env, bk_add)
    bk += [atom_to_str(x) for x in model_fixed]
    asp: Solver = Solver(env.get_ground_atoms(), bk)
    times.record_time("ASP init", time.time() - start_time)

    # Sample initial models
    start_time = time.time()
    spec: Spec = Spec(goal_true=spec_clauses, atoms_true=list(model_fixed), models_banned=models_banned)
    models: List[Model] = asp.get_models(spec, env.on_model, model_batch_size, True)
    times.record_time("Model samp", time.time() - start_time)

    num_models_init: int = len(models)
    num_models_superset: int = 0
    while len(models) > 0:
        if spec_verbose:
            print(f"{len(models)} models")

        # search for goal nodes
        start_time = time.time()
        goal_nodes: List[Optional[Node]] = search_for_goal(env, state_start, models, search_batch_size, weight,
                                                           max_search_itrs, heur_fn, search_verbose, viz_model)

        models_banned += [model for model, goal_node in zip(models, goal_nodes) if goal_node is None]

        models = [model for model, goal_node in zip(models, goal_nodes) if goal_node is not None]
        goal_nodes_found: List[Node] = [goal_node for goal_node in goal_nodes if goal_node is not None]
        if spec_verbose:
            print(f"Found a path to {len(goal_nodes_found)} models")
        times.record_time("Search", time.time() - start_time)

        if len(models) == 0:
            continue

        # check for goal states
        models_terminal: List[Model] = env.state_to_model([goal_node.state for goal_node in goal_nodes_found])
        for goal_node, model_terminal in zip(goal_nodes_found, models_terminal):
            start_time = time.time()
            is_model = asp.check_model(Spec(goal_true=spec_clauses), model_terminal)
            times.record_time("Check", time.time() - start_time)
            if is_model:
                if spec_verbose:
                    print("Found a goal state")
                path_states, path_actions, path_cost = get_path(goal_node)
                return (True, path_states, path_actions, path_cost, num_models_init, num_models_superset, models_banned,
                        times)

        # Get supersets of models
        start_time = time.time()
        samp_per_model: List[int] = misc_utils.split_evenly(model_batch_size, len(models))
        models_superset: List[Model] = []
        for num_samp_i, model in zip(samp_per_model, models):
            spec_sup: Spec = Spec(goal_true=spec_clauses, atoms_true=list(model), models_banned=models_banned,
                                  num_atoms_gt=len(model))
            models_superset += asp.get_models(spec_sup, env.on_model, num_samp_i, True)

        num_models_superset += len(models_superset)
        models = models_superset
        times.record_time("Model superset", time.time() - start_time)

    if spec_verbose:
        print("No path found")

    return False, [], [], -1.0, num_models_init, num_models_superset, models_banned, times
