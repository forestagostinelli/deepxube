""" Definition of State, Action, Goal, and Domain """
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, TypeVar, Generic, Dict, Any

import torch
from torch import nn
import numpy as np
from clingo.solving import Model as ModelCl

from deepxube.logic.logic_objects import Atom, Model
from deepxube.utils import misc_utils
from deepxube.nnet.nnet_utils import NNetPar, NNetCallable, load_nnet
from deepxube.utils.timing_utils import Times

from matplotlib.figure import Figure
import os
import random
import time
from numpy.typing import NDArray


class State(ABC):
    """ State object

    """
    @abstractmethod
    def __hash__(self) -> int:
        """ For use in CLOSED dictionary for pathfinding
        :return: hash value
        """
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """ for use in state reidentification during pathfinding

        :param other: other state
        :return: true if they are equal
        """
        pass


class Action(ABC):
    """ Action object

    """

    @abstractmethod
    def __hash__(self) -> int:
        """ For use in backup for Q* search
        :return: hash value
        """
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """ for use in backup for Q* search

        :param other: other state
        :return: true if they are equal
        """
        pass


class Goal(ABC):
    """ Goal object that represents a set of states considered goal states

    """
    pass


S = TypeVar('S', bound=State)
A = TypeVar('A', bound=Action)
G = TypeVar('G', bound=Goal)


# TODO method for downloading data?
class Domain(ABC, Generic[S, A, G]):
    """ The domain which generates problem instances and defines the relationship between states, actions, and goals
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.nnet_par_dict: Dict[str, Tuple[str, NNetPar]] = dict()
        self.nnet_fn_dict: Dict[str, NNetCallable] = dict()

    @abstractmethod
    def sample_problem_instances(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        """ Return start goal pairs with num_steps_l between start and goal

        :param num_steps_l: Number of steps to take between start and goal. Does not have to directly correspond to number of steps and can also be ignored.
        :param times: Times that can be used to profile code
        :return: List of start states and list of goals
        """
        pass

    @abstractmethod
    def sample_state_action(self, states: List[S]) -> List[A]:
        """ Get a random action that is applicable to the current state

        :param states: List of states
        :return: List of random actions applicable to given states
        """
        pass

    @abstractmethod
    def next_state(self, states: List[S], actions: List[A]) -> Tuple[List[S], List[float]]:
        """ Get the next state and transition cost given the current state and action

        :param states: List of states
        :param actions: List of actions to take
        :return: Next states, transition costs
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[S], goals: List[G]) -> List[bool]:
        """ Returns true if the state is a member of the set of goal states represented by the goal

        :param states: List of states
        :param goals: List of goals
        :return: List of booleans where the element at index i corresponds to whether or not the state at index i is a member of the set of goal states
        represented by the goal at index i
        """
        pass

    def sample_next_state(self, states: List[S]) -> Tuple[List[S], List[A], List[float]]:
        """ Get random next state and transition cost given the current state

        :param states: List of states
        :return: Next states, actions taken, transition costs
        """
        actions_rand: List[A] = self.sample_state_action(states)
        states_next, tcs = self.next_state(states, actions_rand)
        return states_next, actions_rand, tcs

    def random_walk(self, states: List[S], num_steps_l: List[int]) -> Tuple[List[S], List[List[A]], List[float]]:
        """ Perform a random walk on the given states for the given number of steps

        :param states: List of states
        :param num_steps_l: number of steps to take for each state
        :return: The resulting state, actions taken, and the path cost for each random walk
        """
        states_walk: List[S] = [state for state in states]
        actions_l: List[List[A]] = [[] for _ in states]
        path_costs: List[float] = [0.0 for _ in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_steps_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        steps_lt: NDArray[np.bool_] = num_steps_curr < num_steps
        while np.any(steps_lt):
            idxs: NDArray[np.int_] = np.where(steps_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            states_moved, actions, tcs = self.sample_next_state(states_to_move)

            idx: int
            for move_idx, idx in enumerate(idxs):
                states_walk[idx] = states_moved[move_idx]
                actions_l[idx].append(actions[move_idx])
                path_costs[idx] += tcs[move_idx]

            num_steps_curr[idxs] = num_steps_curr[idxs] + 1

            steps_lt[idxs] = num_steps_curr[idxs] < num_steps[idxs]

        return states_walk, actions_l, path_costs

    def get_nnet_par_dict(self) -> Dict[str, Tuple[str, NNetPar]]:
        """

        :return: Copy of dict with names of nnets mapped to their file and NNetPar
        """
        return self.nnet_par_dict.copy()

    def set_nnet_fns(self, nnet_fn_dict: Dict[str, NNetCallable]) -> None:
        """

        :param nnet_fn_dict: A dictionary mapping nnet names to NNetCallable
        :return: None
        """
        for nnet_name, nnet_fn in nnet_fn_dict.items():
            if nnet_name in self.nnet_par_dict.keys():
                self.nnet_fn_dict[nnet_name] = nnet_fn

    def get_nnet_fn(self, nnet_fn_name: str) -> NNetCallable:
        """

        :param nnet_fn_name: Name of nnet
        :return: NNetCallable obtained from self.nnet_fn_dict
        """
        nnet_fn: Optional[NNetCallable] = self.nnet_fn_dict.get(nnet_fn_name)
        if nnet_fn is None:
            device: torch.device = torch.device("cpu")
            if ('CUDA_VISIBLE_DEVICES' in os.environ) and torch.cuda.is_available():
                device = torch.device("cuda:%i" % 0)

            assert nnet_fn_name in self.nnet_par_dict.keys(), f"nnet_fn_name {nnet_fn_name} has not been added to nnet_par_dict"
            nnet_file, nnet_par = self.nnet_par_dict[nnet_fn_name]
            nnet: nn.Module = nnet_par.get_nnet()
            nnet = load_nnet(nnet_file, nnet, device=device)
            nnet.to(device)

            nnet_fn = nnet_par.get_nnet_fn(nnet, None, device, None)

        assert nnet_fn is not None
        return nnet_fn

    def _add_nnet_par(self, nnet_name: str, nnet_file: str, nnet_par: NNetPar) -> None:
        self.nnet_par_dict[nnet_name] = (nnet_file, nnet_par)

    def __getstate__(self) -> Dict:
        self.nnet_fn_dict = dict()
        return self.__dict__

    def __repr__(self) -> str:
        return f"{type(self).__name__}"


# Visualization mixins
class StateGoalVizable(Domain[S, A, G]):
    """ Can visualize problem instances

    """
    @abstractmethod
    def visualize_state_goal(self, state: S, goal: G, fig: Figure) -> None:
        """ Modifies the given figure to visualize the given state and goal

        :param state: State
        :param goal: Goal
        :param fig: Figure to be modified
        """
        pass


class StringToAct(Domain[S, A, G]):
    """ Can get an action from a string. Used when visualizing problem instances.

    """
    @abstractmethod
    def string_to_action(self, act_str: str) -> Optional[A]:
        """
        :param act_str: A string representation of an action
        :return: The action represented by the string, if it is a valid representation, None otherwise
        """
        pass

    @abstractmethod
    def string_to_action_help(self) -> str:
        """
        :return: A description of how actions are represented as strings
        """
        pass


# Action mixins
class ActsFixed(Domain[S, A, G]):
    @abstractmethod
    def sample_action(self, num: int) -> List[A]:
        """ Sample actions
        :param num: number of actions to sample
        :return: Sampled actions
        """
        pass

    def sample_state_action(self, states: List[S]) -> List[A]:
        return self.sample_action(len(states))


class ActsRev(Domain[S, A, G], ABC):
    """ Actions are reversible.

    """
    @abstractmethod
    def sample_rev_state(self, states: List[S]) -> Tuple[List[S], List[A], List[float]]:
        """ Get random reverse state, action that returns reverse state to given state and transition cost along edge going to reverse state

        :param states: List of states
        :return: Reverse states, actions to return to given states, transition cost along edge going to reverse state
        """
        pass


class ActsEnum(Domain[S, A, G]):
    @abstractmethod
    def get_state_actions(self, states: List[S]) -> List[List[A]]:
        """ Get all actions that are applicable to each of the given states

        :param states: List of states
        :return: Applicable actions
        """
        pass

    def sample_state_action(self, states: List[S]) -> List[A]:
        state_actions_l: List[List[A]] = self.get_state_actions(states)
        return [random.choice(state_actions) for state_actions in state_actions_l]

    def expand(self, states: List[S]) -> Tuple[List[List[S]], List[List[A]], List[List[float]]]:
        """ Generate all children for the state, assumes there is at least one child state
        :param states: List of states
        :return: Children of each state, actions, transition costs for each state
        """
        # get actions
        actions_exp_l: List[List[A]] = self.get_state_actions(states)

        # repeat states according to actions
        actions_flat, split_idxs = misc_utils.flatten(actions_exp_l)

        states_flat: List[S] = []
        for state, actions_exp in zip(states, actions_exp_l, strict=True):
            states_flat.extend([state] * len(actions_exp))

        assert len(states_flat) == len(actions_flat), f"{len(states_flat)}, {len(actions_flat)}"

        # get next states
        states_exp_flat, tcs_flat = self.next_state(states_flat, actions_flat)

        # unflatten
        states_exp: List[List[S]] = misc_utils.unflatten(states_exp_flat, split_idxs)
        tcs_l: List[List[float]] = misc_utils.unflatten(tcs_flat, split_idxs)

        return states_exp, actions_exp_l, tcs_l


class ActsEnumFixed(ActsEnum[S, A, G], ActsFixed[S, A, G]):
    def sample_action(self, num: int) -> List[A]:
        actions_fixed: List[A] = self.get_actions_fixed()
        return [random.choice(actions_fixed) for _ in range(num)]

    def get_state_actions(self, states: List[S]) -> List[List[A]]:
        return [self.get_actions_fixed().copy() for _ in range(len(states))]

    @abstractmethod
    def get_actions_fixed(self) -> List[A]:
        """

        :return: All possible actions. Every action should be applicable to any state in the domain.
        """
        pass

    def get_num_acts(self) -> int:
        """

        :return: The number of possible actions
        """
        return len(self.get_actions_fixed())


# supervised data generation
class NodesSupervisable(Domain[S, A, G]):
    @abstractmethod
    def samp_nodes_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[float]]:
        """ Return problem instances with a supervised label for the cost-to-go. This label need not be the true cost-to-go.

        :param steps_gen: Number of actions to take to sample nodes. Labels are the number of actions taken
        :return: States, goals, labels
        """
        pass


class EdgesSupervisable(Domain[S, A, G]):
    @abstractmethod
    def samp_edges_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A], List[float]]:
        """ Return problem instances with a supervised label for the cost-to-go. This label need not be the true cost-to-go.

        :param steps_gen: Number of actions to take to sample nodes. Labels are the number of actions taken
        :return: States, goals, actions, labels
        """
        pass


class EdgesSampleable(Domain[S, A, G]):
    @abstractmethod
    def samp_edges(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A]]:
        """ Sample edges that are on a path to a goal.

        :param steps_gen: Number of steps to take between start state and goal
        :return: States, goals, actions taken from states that lead to goal
        """
        pass


# Goal mixins
class GoalSampleable(Domain[S, A, G]):
    """ Can sample goals """
    @abstractmethod
    def sample_goals(self, num: int) -> List[G]:
        """ Sample goals
        :return: Goals
        """
        pass


class GoalStateSampleable(Domain[S, A, G]):
    """ Can sample goal states """
    @abstractmethod
    def sample_goal_states(self, num: int) -> List[S]:
        """
        :return: Sampled goal states
        """
        pass


class GoalSampleableFromState(Domain[S, A, G]):
    """ Can sample goals from states such that the state is a member of the sampled goal """
    @abstractmethod
    def sample_goal_from_state(self, states_start: Optional[List[S]], states_goal: List[S]) -> List[G]:
        """ Given a state, sample a goal that represents a set of goal states of which the given state is a member.

        :param states_start: Optional list of start states. Can be used to sample goals that are difficult to achieve from the given start state.
        :param states_goal: List of states from which goals will be sampled.
        :return: Goals
        """
        pass


class StateSampleableFromGoal(Domain[S, A, G]):
    """ Can sample states from goals """
    @abstractmethod
    def sample_state_from_goal(self, goals: List[G]) -> List[S]:
        """ Given a goal, sample a state that is a member of the set of states represneted by that goal.

        :param goals: List of goals
        :return: List of list of states, where each state is a member of the set of goal states represented by the corresponding goal
        """
        pass


class GoalFixed(GoalSampleable[S, A, G]):
    """ Goal is the same for all problem instances """

    @abstractmethod
    def get_goal(self) -> G:
        """
        :return: Fixed goal
        """
        pass

    def sample_goals(self, num: int) -> List[G]:
        return [self.get_goal()] * num


class GoalStateGoalPairSampleable(Domain[S, A, G]):
    @abstractmethod
    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[S], List[G]]:
        """

        :param num: Number of state/goal pairs to sample
        :return: pairs of states and corresponding goals of which the sampled state is a member
        """
        pass


class GoalStateSampGoalSamp(GoalStateGoalPairSampleable[S, A, G], GoalStateSampleable[S, A, G], GoalSampleableFromState[S, A, G], ABC):
    """ Sample goal state and then sample goals from goal states """
    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[S], List[G]]:
        states_goal: List[S] = self.sample_goal_states(num)
        goals: List[G] = self.sample_goal_from_state(None, states_goal)
        return states_goal, goals


class GoalSampGoalStateSamp(GoalStateGoalPairSampleable[S, A, G], GoalSampleable[S, A, G], StateSampleableFromGoal[S, A, G], ABC):
    """ Sample goals and then sample goal states from goals """
    def sample_goalstate_goal_pairs(self, num: int) -> Tuple[List[S], List[G]]:
        goals: List[G] = self.sample_goals(num)
        states_goal: List[S] = self.sample_state_from_goal(goals)
        return states_goal, goals


# Problem instance generation mixins
class StartGoalWalkable(GoalSampleableFromState[S, A, G], NodesSupervisable[S, A, G], EdgesSupervisable[S, A, G], EdgesSampleable[S, A, G]):
    """ Can sample start states, take actions to obtain another state, and sample a goal from that state"""
    @abstractmethod
    def sample_start_states(self, num_states: int) -> List[S]:
        """ A method for generating start states. Should try to make this generate states that are as diverse as
        possible so that the trained heuristic function generalizes well.

        :param num_states: Number of states to get
        :return: Generated states
        """
        pass

    def sample_problem_instances(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        # Initialize
        if times is None:
            times = Times()

        # Start states
        start_time = time.time()
        states_start: List[S] = self.sample_start_states(len(num_steps_l))
        times.record_time("sample_start_states", time.time() - start_time)

        # random walk
        start_time = time.time()
        states_goal: List[S] = self.random_walk(states_start, num_steps_l)[0]
        times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[G] = self.sample_goal_from_state(states_start, states_goal)
        times.record_time("sample_goal", time.time() - start_time)

        return states_start, goals

    def samp_nodes_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[float]]:
        states_start: List[S] = self.sample_start_states(len(steps_gen))
        states_goal, _, path_costs = self.random_walk(states_start, steps_gen)
        goals: List[G] = self.sample_goal_from_state(states_start, states_goal)

        return states_start, goals, path_costs

    def samp_edges_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A], List[float]]:
        return self._get_edges_and_labels(steps_gen)

    def samp_edges(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A]]:
        states, goals, actions, _ = self._get_edges_and_labels(steps_gen)
        return states, goals, actions

    def _get_edges_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A], List[float]]:
        # start states
        states_start: List[S] = self.sample_start_states(len(steps_gen))

        # first step
        states_start_1step, actions_init, tcs_step1 = self.sample_next_state(states_start)

        # account for step_gen == 0
        for idx in np.where(np.array(steps_gen) == 0)[0]:
            states_start_1step[idx] = states_start[idx]
            tcs_step1[idx] = 0.0

        # random walk
        steps_gen_minus_1: List[int] = np.maximum(np.array(steps_gen) - 1, 0).tolist()
        states_goal, _, path_costs = self.random_walk(states_start_1step, steps_gen_minus_1)
        path_costs = (np.array(tcs_step1) + np.array(path_costs)).tolist()

        # sample goal
        goals: List[G] = self.sample_goal_from_state(states_start, states_goal)

        return states_start, goals, actions_init, path_costs


class GoalStartRevWalkable(GoalStateGoalPairSampleable[S, A, G]):
    def sample_problem_instances(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        # Initialize
        if times is None:
            times = Times()

        # goals
        start_time = time.time()
        states_goal, goals = self.sample_goalstate_goal_pairs(len(num_steps_l))
        times.record_time("sample_goalstate_goal_pairs", time.time() - start_time)

        # random walk to get start states
        start_time = time.time()
        states_start: List[S] = self.random_walk_rev_no_path_cost(states_goal, num_steps_l)
        times.record_time("random_walk", time.time() - start_time)

        return states_start, goals

    @abstractmethod
    def random_walk_rev_no_path_cost(self, states: List[S], num_steps_l: List[int]) -> List[S]:
        """ Domain need not be reversible as the distance of a path obtained by a number reverse steps can be roughly correlated with the number of steps

        :param states: List of states
        :param num_steps_l: List of integers
        :return: states resulting from reverse random walk
        """
        pass


class GoalStartRevWalkableActsRev(GoalStartRevWalkable[S, A, G], ActsRev[S, A, G], NodesSupervisable[S, A, G], EdgesSupervisable[S, A, G],
                                  EdgesSampleable[S, A, G], ABC):
    def random_walk_rev_no_path_cost(self, states: List[S], num_steps_l: List[int]) -> List[S]:
        return self.random_walk(states, num_steps_l)[0]

    def random_walk_rev(self, states: List[S], num_steps_l: List[int]) -> Tuple[List[S], List[List[A]], List[float]]:
        """ Start from given states and take random walks using reverse actions

        :param states: States
        :param num_steps_l: Number of reverse actions to take for each state
        :return: States along reverse random walk, actions taken along reverse edges, path costs
        """
        states_walk: List[S] = [state for state in states]
        actions_rev_l: List[List[A]] = [[] for _ in states]
        path_costs: List[float] = [0.0 for _ in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_steps_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        steps_lt: NDArray[np.bool_] = num_steps_curr < num_steps
        while np.any(steps_lt):
            idxs: NDArray[np.int_] = np.where(steps_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            states_moved, actions_rev, tcs = self.sample_rev_state(states_to_move)

            idx: int
            for move_idx, idx in enumerate(idxs):
                states_walk[idx] = states_moved[move_idx]
                actions_rev_l[idx].append(actions_rev[move_idx])
                path_costs[idx] += tcs[move_idx]

            num_steps_curr[idxs] = num_steps_curr[idxs] + 1

            steps_lt[idxs] = num_steps_curr[idxs] < num_steps[idxs]

        return states_walk, actions_rev_l, path_costs

    def samp_nodes_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[float]]:
        states_goal, goals = self.sample_goalstate_goal_pairs(len(steps_gen))
        states_start, _, path_costs = self.random_walk_rev(states_goal, steps_gen)

        return states_start, goals, path_costs

    def samp_edges_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A], List[float]]:
        return self._get_edges_and_labels(steps_gen)

    def samp_edges(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A]]:
        states, goals, actions, _ = self._get_edges_and_labels(steps_gen)
        return states, goals, actions

    def _get_edges_and_labels(self, steps_gen: List[int]) -> Tuple[List[S], List[G], List[A], List[float]]:
        # samp_goal_state_goal
        states_goal, goals = self.sample_goalstate_goal_pairs(len(steps_gen))

        # random walk rev
        steps_gen_min_1: List[int] = np.maximum(np.array(steps_gen) - 1, 0).tolist()
        states_start_1step, _, path_costs = self.random_walk_rev(states_goal, steps_gen_min_1)

        # first step
        states_start, actions_init, tcs_step1 = self.sample_rev_state(states_start_1step)

        # account for step_gen == 0
        for idx in np.where(np.array(steps_gen) == 0)[0]:
            states_start[idx] = states_start_1step[idx]
            tcs_step1[idx] = 0.0

        path_costs = (np.array(tcs_step1) + np.array(path_costs)).tolist()

        return states_start, goals, actions_init, path_costs


# numpy convenience mixins
class NextStateNP(Domain[S, A, G]):
    def next_state(self, states: List[S], actions: List[A]) -> Tuple[List[S], List[float]]:
        states_np: List[NDArray] = self._states_to_np(states)
        states_next_np, tcs = self._next_state_np(states_np, actions)
        states_next: List[S] = self._np_to_states(states_next_np)

        return states_next, tcs

    def random_walk(self, states: List[S], num_steps_l: List[int]) -> Tuple[List[S], List[List[A]], List[float]]:
        states_np: List[NDArray] = self._states_to_np(states)
        actions_l: List[List[A]] = [[] for _ in states]
        path_costs: List[float] = [0.0 for _ in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_steps_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        steps_lt: NDArray[np.bool_] = num_steps_curr < num_steps
        while np.any(steps_lt):
            idxs: NDArray[np.int_] = np.where(steps_lt)[0]
            states_np_tomove: List[NDArray] = [states_np_i[idxs] for states_np_i in states_np]
            actions_rand: List[A] = self._sample_state_np_action(states_np_tomove)

            states_moved, tcs = self._next_state_np(states_np_tomove, actions_rand)

            for l_idx in range(len(states_np)):
                states_np[l_idx][idxs] = states_moved[l_idx]

            idx: int
            for act_idx, idx in enumerate(idxs):
                actions_l[idx].append(actions_rand[act_idx])
                path_costs[idx] += tcs[act_idx]

            num_steps_curr[idxs] = num_steps_curr[idxs] + 1

            steps_lt[idxs] = num_steps_curr[idxs] < num_steps[idxs]

        return self._np_to_states(states_np), actions_l, path_costs

    @abstractmethod
    def _states_to_np(self, states: List[S]) -> List[NDArray]:
        pass

    @abstractmethod
    def _np_to_states(self, states_np_l: List[NDArray]) -> List[S]:
        pass

    @abstractmethod
    def _sample_state_np_action(self, states_np: List[NDArray]) -> List[A]:
        pass

    @abstractmethod
    def _next_state_np(self, states_np: List[NDArray], actions: List[A]) -> Tuple[List[NDArray], List[float]]:
        """ Get the next state and transition cost given the current numpy representations of the state and action


        @param states_np: numpy representation of states. Each row in each element of states_np list represents
        information for a different state. There can be one or more multiple elements in the list for each state.
        This object should not be mutated.
        @param actions: actions
        @return: Numpy representation of next states, transition costs
        """
        pass


class NextStateNPActsFixed(NextStateNP[S, A, G], ActsFixed[S, A, G], ABC):
    def _sample_state_np_action(self, states_np: List[NDArray]) -> List[A]:
        return self.sample_action(states_np[0].shape[0])


class NextStateNPActsEnumFixed(NextStateNP[S, A, G], ActsEnumFixed[S, A, G], ABC):
    def _get_state_np_actions(self, states_np: List[NDArray]) -> List[List[A]]:
        state_actions: List[A] = self.get_actions_fixed()
        return [state_actions.copy() for _ in range(states_np[0].shape[0])]

    def _sample_state_np_action(self, states_np: List[NDArray]) -> List[A]:
        state_actions_l: List[List[A]] = self._get_state_np_actions(states_np)
        return [random.choice(state_actions) for state_actions in state_actions_l]


# PDDL Mixins
class SupportsPDDL(Domain[S, A, G], ABC):
    @abstractmethod
    def get_pddl_domain(self) -> List[str]:
        """

        :return: PDDL domain where each entry is a new line
        """
        pass

    @abstractmethod
    def prob_inst_to_pddl_inst(self, state: S, goal: G) -> List[str]:
        """

        :param state: State
        :param goal: Goal
        :return: PDDL problem instance of given state and goal where each entry is a new line
        """
        pass

    @abstractmethod
    def pddl_action_to_action(self, pddl_action: str) -> A:
        """

        :param pddl_action: PDDL action in string representation
        :return: Action
        """
        pass


# Logic mixins
class GoalGrndAtoms(GoalSampleableFromState[S, A, G]):
    @abstractmethod
    def state_to_model(self, states: List[S]) -> List[Model]:
        """

        :param states: States
        :return: Model (set of ground atoms) representation of each state
        """
        pass

    @abstractmethod
    def model_to_state(self, models: List[Model]) -> List[S]:
        """ Assumes model is a fully specified state

        :param models:
        :return:
        """
        pass

    @abstractmethod
    def goal_to_model(self, goals: List[G]) -> List[Model]:
        """

        :param goals: Goals
        :return: Model representation of goals
        """
        pass

    @abstractmethod
    def model_to_goal(self, models: List[Model]) -> List[G]:
        """

        :param models: Models
        :return: Goal representation of models
        """
        pass

    def is_solved(self, states: List[S], goals: List[G]) -> List[bool]:
        """ Returns whether or not state is solved

        :param states: List of states
        :param goals: List of goals
        :return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        models_g: List[Model] = self.goal_to_model(goals)
        is_solved_l: List[bool] = []
        models_s: List[Model] = self.state_to_model(states)
        for model_state, model_goal in zip(models_s, models_g):
            is_solved_l.append(model_goal.issubset(model_state))

        return is_solved_l

    def sample_goal_from_state(self, states_start: Optional[List[S]], states_goal: List[S]) -> List[G]:
        models_g: List[Model] = []

        models_s: List[Model] = self.state_to_model(states_goal)
        keep_probs: NDArray[np.float64] = np.random.rand(len(states_goal))
        for model_s, keep_prob in zip(models_s, keep_probs):
            rand_subset: Set[Atom] = misc_utils.random_subset(model_s, keep_prob)
            models_g.append(frozenset(rand_subset))

        return self.model_to_goal(models_g)

    @abstractmethod
    def get_bk(self) -> List[str]:
        """ get background, each element in list is a line

        :return:
        """
        pass

    @abstractmethod
    def get_ground_atoms(self) -> List[Atom]:
        """ Get all possible ground atoms that can be used to make a state

        :return:
        """
        pass

    @abstractmethod
    def on_model(self, m: ModelCl) -> Model:
        """ Process results from clingo

        :param m:
        :return:
        """
        pass

    @abstractmethod
    def start_state_fixed(self, states: List[S]) -> List[Model]:
        """ Given the start state, what must also be true for the goal state (i.e. immovable walls)

        :param states:
        :return:
        """
        pass
