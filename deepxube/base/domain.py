from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, TypeVar, Generic, Dict, Any
import numpy as np
from clingo.solving import Model as ModelCl

from deepxube.logic.logic_objects import Atom, Model
from deepxube.utils import misc_utils
from deepxube.nnet.nnet_utils import NNetPar, NNetCallable
from deepxube.utils.timing_utils import Times
from matplotlib.figure import Figure
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.nnet_pars: List[Tuple[str, str, NNetPar]] = []

    @abstractmethod
    def get_start_goal_pairs(self, num_steps_l: List[int],
                             times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        """ Return start goal pairs with num_steps_l between start and goal

        :param num_steps_l: Number of steps to take between start and goal
        :param times: Times that can be used to profile code
        :return: List of start states and list of goals
        """
        pass

    @abstractmethod
    def get_state_action_rand(self, states: List[S]) -> List[A]:
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

    def next_state_rand(self, states: List[S]) -> Tuple[List[S], List[float]]:
        """ Get random next state and transition cost given the current state

        :param states: List of states
        :return: Next states, transition costs
        """
        actions_rand: List[A] = self.get_state_action_rand(states)
        return self.next_state(states, actions_rand)

    def random_walk(self, states: List[S], num_steps_l: List[int]) -> Tuple[List[S], List[float]]:
        """ Perform a random walk on the given states for the given number of steps

        :param states: List of states
        :param num_steps_l: number of steps to take for each state
        :return: The resulting state and the path cost for each random walk
        """
        states_walk: List[S] = [state for state in states]
        path_costs: List[float] = [0.0 for _ in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_steps_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        steps_lt: NDArray[np.bool_] = num_steps_curr < num_steps
        while np.any(steps_lt):
            idxs: NDArray[np.int_] = np.where(steps_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            states_moved, tcs = self.next_state_rand(states_to_move)

            idx: int
            for move_idx, idx in enumerate(idxs):
                states_walk[idx] = states_moved[move_idx]
                path_costs[idx] += tcs[move_idx]

            num_steps_curr[idxs] = num_steps_curr[idxs] + 1

            steps_lt[idxs] = num_steps_curr[idxs] < num_steps[idxs]

        return states_walk, path_costs

    def get_nnet_pars(self) -> List[Tuple[str, str, NNetPar]]:
        return self.nnet_pars

    def set_nnet_fns(self, nnet_fn_dict: Dict[str, NNetCallable]) -> None:
        pass


# Visualization Mixins
class StateGoalVizable(Domain[S, A, G]):
    """ Can visualize problem instances

    """
    @abstractmethod
    def visualize_state_goal(self, state: S, goal: G, fig: Figure) -> None:
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


class ActsFixed(Domain[S, A, G]):
    @abstractmethod
    def get_action_rand(self, num: int) -> List[A]:
        pass

    def get_state_action_rand(self, states: List[S]) -> List[A]:
        return self.get_action_rand(len(states))


class ActsRev(Domain[S, A, G], ABC):
    """ Actions are reversible.

    """
    @abstractmethod
    def rev_action(self, actions: List[A]) -> List[A]:
        """ Get the reverse of the given action

        :param actions: List of actions
        :return: Reverse of given action
        """
        pass

    @abstractmethod
    def rev_state(self, states: List[S], actions: List[A]) -> Tuple[List[S], List[float]]:
        """ Transition along the directed edge in the reverse direction.

        :param states: List of states
        :param actions: List of actions to take
        :return: Reverse states, transition costs which are weights of edges taken in reverse
        """
        pass


class ActsEnum(Domain[S, A, G]):
    @abstractmethod
    def get_state_actions(self, states: List[S]) -> List[List[A]]:
        """ Get actions applicable to each states

        :param states: List of states
        :return: Applicable actions
        """
        pass

    def get_state_action_rand(self, states: List[S]) -> List[A]:
        state_actions_l: List[List[A]] = self.get_state_actions(states)
        return [random.choice(state_actions) for state_actions in state_actions_l]

    def expand(self, states: List[S]) -> Tuple[List[List[S]], List[List[A]], List[List[float]]]:
        """ Generate all children for the state, assumes there is at least one child state
        :param states: List of states
        :return: Children of each state, actions, transition costs for each state
        """
        # TODO further validate
        # initialize
        states_exp_l: List[List[S]] = [[] for _ in range(len(states))]
        actions_exp_l: List[List[A]] = [[] for _ in range(len(states))]
        tcs_l: List[List[float]] = [[] for _ in range(len(states))]
        state_actions: List[List[A]] = self.get_state_actions(states)

        num_actions_tot: NDArray[np.int_] = np.array([len(x) for x in state_actions])
        num_actions_taken: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        actions_lt: NDArray[np.bool_] = num_actions_taken < num_actions_tot

        # for each move, get next states, transition costs, and if solved
        while np.any(actions_lt):
            idxs: NDArray[np.int_] = np.where(actions_lt)[0]
            states_idxs: List[S] = [states[idx] for idx in idxs]
            actions_idxs: List[A] = [state_actions[idx].pop(0) for idx in idxs]

            # next state
            states_next, tcs_move = self.next_state(states_idxs, actions_idxs)

            # append
            idx: int
            for exp_idx, idx in enumerate(idxs):
                states_exp_l[idx].append(states_next[exp_idx])
                actions_exp_l[idx].append(actions_idxs[exp_idx])
                tcs_l[idx].append(tcs_move[exp_idx])

            num_actions_taken[idxs] = num_actions_taken[idxs] + 1
            actions_lt[idxs] = num_actions_taken[idxs] < num_actions_tot[idxs]

        return states_exp_l, actions_exp_l, tcs_l


class ActsEnumFixed(ActsEnum[S, A, G], ActsFixed[S, A, G]):
    def get_action_rand(self, num: int) -> List[A]:
        actions_fixed: List[A] = self.get_actions_fixed()
        return [random.choice(actions_fixed) for _ in range(num)]

    def get_state_actions(self, states: List[S]) -> List[List[A]]:
        return [self.get_actions_fixed().copy() for _ in range(len(states))]

    @abstractmethod
    def get_actions_fixed(self) -> List[A]:
        pass

    def get_num_acts(self) -> int:
        return len(self.get_actions_fixed())


# Goal mixins
class GoalSampleable(Domain[S, A, G]):
    """ Can sample goals """
    @abstractmethod
    def sample_goals(self, num: int) -> List[G]:
        """ Sample goals
        :return: Goals
        """
        pass


class GoalSampleableFromState(Domain[S, A, G]):
    """ Can sample goals from states such that the state is a member of the sampled goal """
    @abstractmethod
    def sample_goal_from_state(self, states_start: List[S], states_goal: List[S]) -> List[G]:
        """ Given a state, sample a goal that represents a set of goal states of which the given state is a member.

        :param states_start: List of start states. Can be used to sample goals that are difficult to achieve from the given start state.
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


# Problem instance mixins
class StartGoalWalkable(GoalSampleableFromState[S, A, G]):
    """ Can sample start states, take actions to obtain another state, and sample a goal from that state"""
    @abstractmethod
    def get_start_states(self, num_states: int) -> List[S]:
        """ A method for generating start states. Should try to make this generate states that are as diverse as
        possible so that the trained heuristic function generalizes well.

        :param num_states: Number of states to get
        :return: Generated states
        """
        pass

    def get_start_goal_pairs(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        # Initialize
        if times is None:
            times = Times()

        # Start states
        start_time = time.time()
        states_start: List[S] = self.get_start_states(len(num_steps_l))
        times.record_time("get_start_states", time.time() - start_time)

        # random walk
        start_time = time.time()
        states_goal: List[S] = self.random_walk(states_start, num_steps_l)[0]
        times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[G] = self.sample_goal_from_state(states_start, states_goal)
        times.record_time("sample_goal", time.time() - start_time)

        return states_start, goals


class GoalStateRevWalkable(GoalSampleable[S, A, G], StateSampleableFromGoal[S, A, G]):
    def get_start_goal_pairs(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        # Initialize
        if times is None:
            times = Times()

        # goals
        start_time = time.time()
        goals: List[G] = self.sample_goals(len(num_steps_l))
        times.record_time("sample_goal", time.time() - start_time)

        # goal states
        start_time = time.time()
        states_goal: List[S] = self.sample_state_from_goal(goals)
        times.record_time("get_goal_states", time.time() - start_time)

        # random walk to get start states
        start_time = time.time()
        states_start: List[S] = self.random_walk_rev(states_goal, num_steps_l)
        times.record_time("random_walk", time.time() - start_time)

        return states_start, goals

    @abstractmethod
    def random_walk_rev(self, states: List[S], num_steps_l: List[int]) -> List[S]:
        pass



class FixedGoalRevWalkActsRev(GoalStateRevWalkable[S, A, G], ActsRev[S, A, G], ABC):
    def random_walk_rev(self, states: List[S], num_steps_l: List[int]) -> List[S]:
        return self.random_walk(states, num_steps_l)[0]


# numpy convenience mixins
class NextStateNP(Domain[S, A, G]):
    def next_state(self, states: List[S], actions: List[A]) -> Tuple[List[S], List[float]]:
        states_np: List[NDArray] = self._states_to_np(states)
        states_next_np, tcs = self._next_state_np(states_np, actions)
        states_next: List[S] = self._np_to_states(states_next_np)

        return states_next, tcs

    def random_walk(self, states: List[S], num_steps_l: List[int]) -> Tuple[List[S], List[float]]:
        states_np = self._states_to_np(states)
        path_costs: List[float] = [0.0 for _ in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_steps_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        steps_lt: NDArray[np.bool_] = num_steps_curr < num_steps
        while np.any(steps_lt):
            idxs: NDArray[np.int_] = np.where(steps_lt)[0]
            states_np_tomove: List[NDArray] = [states_np_i[idxs] for states_np_i in states_np]
            actions_rand: List[A] = self._get_state_np_action_rand(states_np_tomove)

            states_moved, tcs = self._next_state_np(states_np_tomove, actions_rand)

            for l_idx in range(len(states_np)):
                states_np[l_idx][idxs] = states_moved[l_idx]
            idx: int
            for act_idx, idx in enumerate(idxs):
                path_costs[idx] += tcs[act_idx]

            num_steps_curr[idxs] = num_steps_curr[idxs] + 1

            steps_lt[idxs] = num_steps_curr[idxs] < num_steps[idxs]

        return self._np_to_states(states_np), path_costs

    @abstractmethod
    def _states_to_np(self, states: List[S]) -> List[NDArray]:
        pass

    @abstractmethod
    def _np_to_states(self, states_np_l: List[NDArray]) -> List[S]:
        pass

    @abstractmethod
    def _get_state_np_actions(self, states_np_l: List[NDArray]) -> List[List[A]]:
        pass

    def _get_state_np_action_rand(self, states_np: List[NDArray]) -> List[A]:
        state_actions_l: List[List[A]] = self._get_state_np_actions(states_np)
        return [random.choice(state_actions) for state_actions in state_actions_l]

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


class NextStateNPActsEnum(NextStateNP[S, A, G], ActsEnum[S, A, G], ABC):
    def expand(self, states: List[S]) -> Tuple[List[List[S]], List[List[A]], List[List[float]]]:
        # initialize
        states_np: List[NDArray] = self._states_to_np(states)
        states_exp_l: List[List[S]] = [[] for _ in range(len(states))]
        actions_exp_l: List[List[A]] = [[] for _ in range(len(states))]
        tcs_l: List[List[float]] = [[] for _ in range(len(states))]
        state_actions: List[List[A]] = self.get_state_actions(states)

        num_actions_tot: NDArray[np.int_] = np.array([len(x) for x in state_actions])
        num_actions_taken: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        actions_lt: NDArray[np.bool_] = num_actions_taken < num_actions_tot

        # for each move, get next states, transition costs, and if solved
        while np.any(actions_lt):
            idxs: NDArray[np.int_] = np.where(actions_lt)[0]
            states_np_idxs: List[NDArray] = [states_np_i[idxs] for states_np_i in states_np]
            actions_idxs: List[A] = [state_actions[idx].pop(0) for idx in idxs]

            # next state
            states_next_np, tcs_move = self._next_state_np(states_np_idxs, actions_idxs)
            states_next: List[S] = self._np_to_states(states_next_np)

            # append
            idx: int
            for exp_idx, idx in enumerate(idxs):
                states_exp_l[idx].append(states_next[exp_idx])
                actions_exp_l[idx].append(actions_idxs[exp_idx])
                tcs_l[idx].append(tcs_move[exp_idx])

            num_actions_taken[idxs] = num_actions_taken[idxs] + 1
            actions_lt[idxs] = num_actions_taken[idxs] < num_actions_tot[idxs]

        return states_exp_l, actions_exp_l, tcs_l


class NextStateNPActsEnumFixed(NextStateNPActsEnum[S, A, G], ActsEnumFixed[S, A, G], ABC):
    def _get_state_np_actions(self, states_np: List[NDArray]) -> List[List[A]]:
        state_actions: List[A] = self.get_actions_fixed()
        return [state_actions.copy() for _ in range(states_np[0].shape[0])]


class SupportsPDDL(Domain[S, A, G], ABC):
    @abstractmethod
    def get_pddl_domain(self) -> List[str]:
        pass

    @abstractmethod
    def state_goal_to_pddl_inst(self, state: S, goal: G) -> List[str]:
        pass

    @abstractmethod
    def pddl_action_to_action(self, pddl_action: str) -> A:
        pass


class GoalGrndAtoms(GoalSampleableFromState[S, A, G]):
    @abstractmethod
    def state_to_model(self, states: List[S]) -> List[Model]:
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
        pass

    @abstractmethod
    def model_to_goal(self, models: List[Model]) -> List[G]:
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

    def sample_goal_from_state(self, states_start: List[S], states_goal: List[S]) -> List[G]:
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
