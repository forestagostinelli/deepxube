from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, TypeVar, Generic, Protocol, runtime_checkable, Union, Dict
import numpy as np
from clingo.solving import Model as ModelCl

from deepxube.logic.logic_objects import Atom, Model
from deepxube.utils import misc_utils
from deepxube.nnet.nnet_utils import NNetPar, NNetCallable
from deepxube.utils.timing_utils import Times
import random
import time
from numpy.typing import NDArray


class State(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        """ For use in CLOSED dictionary for pathfinding
        @return: hash value
        """
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """ for use in state reidentification during pathfinding

        @param other: other state
        @return: true if they are equal
        """
        pass


class Action(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        """ For use in backup for Q* search
        @return: hash value
        """
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """ for use in backup for Q* search

        @param other: other state
        @return: true if they are equal
        """
        pass


# TODO make hash and eq method for goal required
class Goal(ABC):
    pass


S = TypeVar('S', bound=State)
A = TypeVar('A', bound=Action)
G = TypeVar('G', bound=Goal)


class Env(ABC, Generic[S, A, G]):
    def __init__(self) -> None:
        self.nnet_pars: List[Tuple[str, str, NNetPar]] = []

    @abstractmethod
    def get_start_goal_pairs(self, num_steps_l: List[int],
                             times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        """ Return start goal pairs with num_steps_l between start and goal

        @param num_steps_l: Number of steps to take between start and goal
        @param times: Times that can be used to profile code
        @return: List of start states and list of goals
        """
        pass

    @abstractmethod
    def get_state_action_rand(self, states: List[S]) -> List[A]:
        """ Get a random action that is applicable to the current state

        @param states: List of states
        @return: Applicable actions
        """
        pass

    @abstractmethod
    def next_state(self, states: List[S], actions: List[A]) -> Tuple[List[S], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param actions: List of actions to take
        @return: Next states, transition costs
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[S], goals: List[G]) -> List[bool]:
        """ Returns true if the state is a member of the set of goal states represented by the goal

        @param states: List of states
        @param goals: List of goals
        @return: List of booleans where the element at index i corresponds to whether or not the
        state at index i is a member of the set of goal states represented by the goal at index i
        """
        pass

    def get_nnet_pars(self) -> List[Tuple[str, str, NNetPar]]:
        return self.nnet_pars

    def set_nnet_fns(self, nnet_fn_dict: Dict[str, NNetCallable]) -> None:
        pass

    def next_state_rand(self, states: List[S]) -> Tuple[List[S], List[float]]:
        """ Get random next state and transition cost given the current state

        @param states: List of states
        @return: Next states, transition costs
        """
        actions_rand: List[A] = self.get_state_action_rand(states)
        return self.next_state(states, actions_rand)

    def random_walk(self, states: List[S], num_steps_l: List[int]) -> List[S]:
        states_walk: List[S] = [state for state in states]

        num_steps: NDArray[np.int_] = np.array(num_steps_l)
        num_moves_curr: NDArray[np.int_] = np.zeros(len(states), dtype=int)
        moves_lt: NDArray[np.bool_] = num_moves_curr < num_steps
        while np.any(moves_lt):
            idxs: NDArray[np.int_] = np.where(moves_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            states_moved, _ = self.next_state_rand(states_to_move)

            idx: int
            for move_idx, idx in enumerate(idxs):
                states_walk[idx] = states_moved[move_idx]

            num_moves_curr[idxs] = num_moves_curr[idxs] + 1

            moves_lt[idxs] = num_moves_curr[idxs] < num_steps[idxs]

        return states_walk


# Mixins
class EnvEnumerableActs(Env[S, A, G]):
    @abstractmethod
    def get_state_actions(self, states: List[S]) -> List[List[A]]:
        """ Get actions applicable to each states

        @param states: List of states
        @return: Applicable actions
        """
        pass

    def get_state_action_rand(self, states: List[S]) -> List[A]:
        state_actions_l: List[List[A]] = self.get_state_actions(states)
        return [random.choice(state_actions) for state_actions in state_actions_l]

    def expand(self, states: List[S]) -> Tuple[List[List[S]], List[List[A]], List[List[float]]]:
        """ Generate all children for the state, assumes there is at least one child state
        @param states: List of states
        @return: Children of each state, actions, transition costs for each state
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

            # transition cost
            idx: int
            for exp_idx, idx in enumerate(idxs):
                states_exp_l[idx].append(states_next[exp_idx])
                actions_exp_l[idx].append(actions_idxs[exp_idx])
                tcs_l[idx].append(tcs_move[exp_idx])

            num_actions_taken[idxs] = num_actions_taken[idxs] + 1
            actions_lt[idxs] = num_actions_taken[idxs] < num_actions_tot[idxs]

        return states_exp_l, actions_exp_l, tcs_l


class EnvGoalSampleable(Env[S, A, G]):
    @abstractmethod
    def sample_goal(self, states_start: List[S], states_goal: List[S]) -> List[G]:
        """ Given a state, return a goal that represents a set of goal states of which the given state is a member.
        Does not have to always return the same goal.

        @param states_start: List of start states
        @param states_goal List of states from which goals will be sampled
        @return: Goals
        """
        pass


class EnvStartGoalRW(EnvGoalSampleable[S, A, G]):
    @abstractmethod
    def get_start_states(self, num_states: int) -> List[S]:
        """ A method for generating start states. Should try to make this generate states that are as diverse as
        possible so that the trained heuristic function generalizes well.

        @param num_states: Number of states to get
        @return: Generated states
        """
        pass

    def get_start_goal_pairs(self, num_steps_l: List[int],
                             times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        # Initialize
        if times is None:
            times = Times()

        # Start states
        start_time = time.time()
        states_start: List[S] = self.get_start_states(len(num_steps_l))
        times.record_time("get_start_states", time.time() - start_time)

        # random walk
        start_time = time.time()
        states_goal: List[S] = self.random_walk(states_start, num_steps_l)
        times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[G] = self.sample_goal(states_start, states_goal)
        times.record_time("sample_goal", time.time() - start_time)

        return states_start, goals


class EnvGrndAtoms(Env[S, A, G]):
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

        @param states: List of states
        @param goals: List of goals
        @return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        models_g: List[Model] = self.goal_to_model(goals)
        is_solved_l: List[bool] = []
        models_s: List[Model] = self.state_to_model(states)
        for model_state, model_goal in zip(models_s, models_g):
            is_solved_l.append(model_goal.issubset(model_state))

        return is_solved_l

    def sample_goal(self, states_start: List[S], states_goal: List[S]) -> List[G]:
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


# Protocols
@runtime_checkable
class EnvVizable(Protocol):
    def visualize(self, states: List[Union[State, Goal]]) -> NDArray[np.float64]: ...


@runtime_checkable
class EnvSupportsPDDL(Protocol):
    def get_pddl_domain(self) -> List[str]: ...

    def state_goal_to_pddl_inst(self, state: State, goal: Goal) -> List[str]: ...

    def pddl_action_to_action(self, pddl_action: str) -> Action: ...
