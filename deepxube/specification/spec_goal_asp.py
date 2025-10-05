from typing import Optional, List, Tuple, Set, Callable, cast
from collections import OrderedDict
from deepxube.base.environment import EnvGrndAtoms, State, Goal, Action
from deepxube.logic.asp import Spec
from deepxube.logic.logic_objects import Clause, Model, Atom
from deepxube.logic.logic_utils import atom_to_str
from deepxube.logic.asp import Solver
from deepxube.utils import viz_utils
from deepxube.utils.timing_utils import Times
from heapq import heappush, heappop
import numpy as np
import random
import time
from dataclasses import dataclass


@dataclass
class PathSoln:
    path_states: List[State]
    path_actions: List[Action]
    path_cost: float


PathFn = Callable[[EnvGrndAtoms, List[State], List[Goal]], List[Optional[PathSoln]]]


class MCArgs:
    def __init__(self, expand_size: int, temp: float, prior: int):
        self.expand_size: int = expand_size
        self.temp: float = temp
        self.prior: int = prior


class RefineArgs:
    def __init__(self, rand: bool, conf: bool, conf_keep: bool, refine_size: int):
        self.rand: bool = rand
        self.conf: bool = conf
        self.conf_keep: bool = conf_keep
        self.refine_size: int = refine_size


class SpecNode:
    def __init__(self, model: Optional[Model], conflict: Model, lb: float):
        self.model: Optional[Model] = model
        self.conflict: Model = conflict
        self.lb: float = lb
        self.num_seen: int = 0
        self.models_prev: List[Model] = []


def get_bk(env: EnvGrndAtoms, bk_add_file_name: Optional[str]) -> List[str]:
    bk_init: List[str] = env.get_bk()
    bk_init.append("")

    if bk_add_file_name is not None:
        bk_add_file = open(bk_add_file_name, 'r')
        bk_init.extend(bk_add_file.read().split("\n"))
        bk_add_file.close()

    return bk_init


class SpecSearchASP:
    def __init__(self, env: EnvGrndAtoms, state_start: State, spec_clauses: List[Clause], path_fn: PathFn,
                 refine_args: RefineArgs, bk_add: Optional[str] = None, verbose: bool = False,
                 viz_model: bool = False, viz_conf: bool = False, viz_reached: bool = False):
        """
        :param env: EnvGrndAtoms environment
        :param state_start: Starting state
        :param spec_clauses: Clauses for specification. All must have goal in the head.
        :param path_fn: Pathfinding function
        :param refine_args: Arguments for refining models of which non-goal states are members
        :param bk_add: A file for additional background information
        :param verbose: Verbose monte carlo pathfinding if true
        :param viz_model: Set true to visualize each model before pathfinding
        :param viz_conf: Set true to visualize each conflict
        :param viz_reached: Set true to visualize reached states
        """
        # Init
        self.times: Times = Times(time_names=["init", "pop", "refine", "check_goal", "path_find", "get_conflict",
                                              "improve_ub", "push"])
        self.models_banned: List[Model] = []

        self.env: EnvGrndAtoms = env
        self.state_start: State = state_start
        self.model_fixed: Model = self.env.start_state_fixed([self.state_start])[0]
        self.path_fn: PathFn = path_fn

        self.spec_clauses: List[Clause] = spec_clauses

        self.refine_args: RefineArgs = refine_args

        self.verbose: bool = verbose

        self.viz_model: bool = viz_model
        self.viz_reached: bool = viz_reached
        self.viz_conf: bool = viz_conf

        # Initialize ASP
        start_time = time.time()
        self.asp: Solver = self._init_asp(bk_add)
        self.times.record_time("init", time.time() - start_time)

        # initialize pathfinding
        self.ub: float = np.inf
        self.specnode_q: List[Tuple[float, int, float, int, SpecNode]] = []
        self.heappush_count: int = 0
        self.models_seen: Set[Model] = set()
        self.num_models_gen: int = 0

        self.soln_best: Optional[PathSoln] = None
        self.stats: List[OrderedDict] = []

        self._push_to_q([SpecNode(None, frozenset(), 0.0)])

        self.step_itr: int = 0

    def step(self):
        times_i = Times()

        # pop
        start_time = time.time()
        node_popped: SpecNode = heappop(self.specnode_q)[4]
        if self.viz_model and (node_popped.model is not None):
            print("Popped")
            viz_utils.visualize_examples(self.env, self.env.model_to_goal([node_popped.model]))

        times_i.record_time("pop", time.time() - start_time)

        # get specializations
        start_time = time.time()
        models_refined: List[Model] = self._refine(node_popped)
        if len(models_refined) >= self.refine_args.refine_size:
            node_popped.num_seen = node_popped.num_seen + 1
            node_popped.models_prev.extend(models_refined)
            self._push_to_q([node_popped])
        num_refined: int = len(models_refined)
        times_i.record_time("refine", time.time() - start_time)

        # check seen
        start_time = time.time()
        models_refined = self._check_seen(models_refined)
        num_not_seen: int = len(models_refined)
        times_i.record_time("check_seen", time.time() - start_time)

        # try to find a path
        start_time = time.time()
        path_solns, states_term_m = self._find_path(models_refined)
        num_reached: int = len([path_soln for path_soln in path_solns if path_soln is not None])
        times_i.record_time("path_find", time.time() - start_time)

        # check if terminal states are goal states
        start_time = time.time()
        is_goal_l: List[bool] = self._is_goal(states_term_m)
        num_reached_not_goal: int = len([path_soln for path_soln, is_goal in zip(path_solns, is_goal_l)
                                         if (path_soln is not None) and (not is_goal)])
        times_i.record_time("check_goal", time.time() - start_time)

        spec_nodes_to_push: List[SpecNode] = []
        for model, path_soln, state_term_m, is_goal in zip(models_refined, path_solns, states_term_m, is_goal_l):
            # visualize
            if self.viz_model:
                print("Specialization")
                viz_utils.visualize_examples(self.env, self.env.model_to_goal([model]))
            if (path_soln is not None) and (path_soln.path_cost < self.ub) and (not is_goal):
                start_time = time.time()
                assert state_term_m is not None
                conflict: Model = self._get_conflict([state_term_m])[0]
                spec_nodes_to_push.append(SpecNode(model, conflict, path_soln.path_cost))
                if self.viz_reached:
                    print("Reached not goal state")
                    viz_utils.visualize_examples(self.env, [path_soln.path_states[-1]])

                if self.viz_conf:
                    print("Conflict")
                    viz_utils.visualize_examples(self.env, self.env.model_to_goal([conflict]))
                times_i.record_time("get_conflict", time.time() - start_time)
            else:
                self.models_banned.append(model)
                if (path_soln is not None) and (path_soln.path_cost < self.ub) and is_goal:
                    start_time = time.time()
                    # get path
                    self.ub = path_soln.path_cost
                    self.soln_best = path_soln

                    # only keep if lb < ub
                    specnode_q_prev: List[Tuple[float, int, float, int, SpecNode]] = self.specnode_q
                    self.specnode_q = []
                    for elem in specnode_q_prev:
                        spec_node: SpecNode = elem[-1]
                        if spec_node.lb < self.ub:
                            self._push_to_q([spec_node])
                        else:
                            assert spec_node.model is not None
                            self.models_banned.append(spec_node.model)

                    if self.viz_reached:
                        print("Reached goal state")
                        viz_utils.visualize_examples(self.env, [path_soln.path_states[-1]])

                    times_i.record_time("improve_ub", time.time() - start_time)

        # push
        start_time = time.time()
        self._push_to_q(spec_nodes_to_push)
        times_i.record_time("get_conflict", time.time() - start_time)

        # updater states
        stats_itr: OrderedDict = OrderedDict([("itr", self.step_itr), ("ub", self.ub),
                                              ("#refined", num_refined), ("#not_seen", num_not_seen),
                                              ("#reached", num_reached), ("#reached_not_goal", num_reached_not_goal),
                                              ("#pushed", len(spec_nodes_to_push)),
                                              ("q_size", len(self.specnode_q))])
        self.stats.append(stats_itr)
        self.times.add_times(times_i)

        # verbose
        if self.verbose:
            lb_min: float = np.inf
            lb_max: float = np.inf
            if len(spec_nodes_to_push) > 0:
                lbs: List[float] = [node.lb for node in spec_nodes_to_push]
                lb_min = min(lbs)
                lb_max = max(lbs)
            stats_str: str = ", ".join(f"{key}: {val}" for key, val in stats_itr.items())
            stats_str = f"{stats_str}, lb(min/max): {lb_min:.2f}/{lb_max:.2f}"
            print(stats_str)
            print(f"Times - {times_i.get_time_str()}")

        self.step_itr += 1

    def is_terminal(self) -> bool:
        return len(self.specnode_q) == 0

    def _init_asp(self, bk_add: Optional[str]) -> Solver:
        bk: List[str] = get_bk(self.env, bk_add)
        bk += [atom_to_str(x) for x in self.model_fixed]  # TODO necessary?
        return Solver(self.env.get_ground_atoms(), bk)

    def _refine(self, node: SpecNode) -> List[Model]:
        # get specs
        specs: List[Spec] = self._get_refine_specs(node.model, node.conflict)

        # get refined models
        models_refined: List[Model] = []
        spec_idxs: List[int] = list(range(len(specs)))
        while (len(models_refined) < self.refine_args.refine_size) and (len(spec_idxs) > 0):
            random.shuffle(spec_idxs)
            spec_idx = spec_idxs.pop(0)
            spec = specs[spec_idx]

            spec_ban_prev: Spec = Spec(models_banned=models_refined + node.models_prev)
            spec_add_ban_prev: Spec = spec.add(spec_ban_prev)

            models_i: List[Model] = self.asp.get_models(spec_add_ban_prev, self.env.on_model, 1, True)
            models_refined += models_i
            if len(models_i) > 0:
                spec_idxs.append(spec_idx)

        self.num_models_gen += len(models_refined)
        return models_refined

    def _get_refine_specs(self, model: Optional[Model], conflict: Model) -> List[Spec]:
        num_atoms_gt: Optional[int]
        atoms_true: Optional[List[Atom]]
        if model is not None:
            atoms_true = list(model)
            num_atoms_gt = len(model)
        else:
            atoms_true = None
            num_atoms_gt = None

        specs_refine: List[Spec] = []
        if self.refine_args.rand or (model is None):
            specs_refine.append(Spec(goal_true=self.spec_clauses, atoms_true=atoms_true,
                                     models_banned=self.models_banned, num_atoms_gt=num_atoms_gt))

        if (self.refine_args.conf or self.refine_args.conf_keep) and (model is not None):
            assert atoms_true is not None
            if self.refine_args.conf:
                # make single atom from model explanation not true via classical negation
                for atom in conflict:
                    atom_cneg: Atom = (f"-{atom[0]}",) + atom[1:]
                    specs_refine.append(Spec(goal_true=self.spec_clauses, atoms_true=atoms_true + [atom_cneg],
                                             models_banned=self.models_banned, num_atoms_gt=num_atoms_gt))

            if self.refine_args.conf_keep:
                # spec for larger models with explanation true
                specs_refine.append(Spec(goal_true=self.spec_clauses, atoms_true=atoms_true + list(conflict),
                                         models_banned=self.models_banned, num_atoms_gt=num_atoms_gt))

        return specs_refine

    def _check_seen(self, models: List[Model]) -> List[Model]:
        models = list(set(models))

        models_not_seen: List[Model] = []
        for model in models:
            if model not in self.models_seen:
                models_not_seen.append(model)
                self.models_seen.add(model)

        return models_not_seen

    def _find_path(self, models: List[Model]) -> Tuple[List[Optional[PathSoln]], List[Optional[Model]]]:
        # check for empty
        if len(models) == 0:
            return [], []

        # get goals from models
        goals: List[Goal] = self.env.model_to_goal(models)

        # initialize
        paths_solns: List[Optional[PathSoln]] = self.path_fn(self.env, [self.state_start] * len(goals), goals)

        # get models of terminal states
        states_term_m: List[Optional[Model]] = [None] * len(paths_solns)
        reached_idxs: List[int] = [idx for idx in range(len(paths_solns)) if paths_solns[idx] is not None]

        if len(reached_idxs) > 0:
            path_solns_reached: List[PathSoln] = cast(List[PathSoln], [paths_solns[idx] for idx in reached_idxs])
            states_term_m_reached: List[Model] = self.env.state_to_model([path_soln.path_states[-1]
                                                                          for path_soln in path_solns_reached])
            for reached_idx, state_term_m_reached in zip(reached_idxs, states_term_m_reached):
                states_term_m[reached_idx] = state_term_m_reached

        return paths_solns, states_term_m

    def _is_goal(self, states_m: List[Optional[Model]]) -> List[bool]:
        # if empty
        if len(states_m) == 0:
            return []

        spec: Spec = Spec(goal_true=self.spec_clauses)
        is_goal_l: List[bool] = [self.asp.check_model(spec, state_m) if state_m is not None else False
                                 for state_m in states_m]
        return is_goal_l

    def _get_conflict(self, states_term_m: List[Model]) -> List[Model]:
        if len(states_term_m) == 0:
            return []

        spec_failure: Spec = Spec(goal_false=self.spec_clauses, atoms_true=list(self.model_fixed))

        models_conflict: List[Model] = []
        for state_term_m in states_term_m:
            model_conflict: Model = self.asp.sample_minimal_model(spec_failure, state_term_m, self.env.on_model)
            models_conflict.append(model_conflict)

        return models_conflict

    def _push_to_q(self, spec_nodes: List[SpecNode]):
        for spec_node in spec_nodes:
            priority: float
            if spec_node.model is None:
                priority = 0
            else:
                priority = -len(spec_node.model)
            heappush(self.specnode_q, (priority, spec_node.num_seen, spec_node.lb, self.heappush_count,
                                       spec_node))
            self.heappush_count += 1
