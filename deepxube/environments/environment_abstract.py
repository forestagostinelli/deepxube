from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Set, Any
import numpy as np
import torch.nn as nn
from deepxube.logic.program import Atom, Model
from deepxube.utils import misc_utils
from deepxube.utils.timing_utils import Times
import random
import time


class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Goal(ABC):
    pass


class HeurFnNNet(nn.Module):
    valid_nnet_types: Set[str] = {"V", "Q"}

    def __init__(self, nnet_type: str):
        super().__init__()
        nnet_type = nnet_type.upper()
        assert nnet_type in self.valid_nnet_types, (f"Invalid nnet type: {nnet_type}. Valid nnet types are: "
                                                    f"{self.valid_nnet_types}")
        self.nnet_type = nnet_type


class Environment(ABC):
    def __init__(self, env_name: str):
        self.env_name: Optional[str] = env_name

    @abstractmethod
    def next_state(self, states: List[State], actions: List[Any]) -> Tuple[List[State], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param actions: Actions to take
        @return: Next states, transition costs
        """
        pass

    def next_state_rand(self, states: List[State]) -> Tuple[List[State], List[float]]:
        """ Get random next state and transition cost given the current state

        @param states: List of states
        @return: Next states, transition costs
        """
        state_actions: List[List[int]] = self.get_state_actions(states)
        actions_rand: List[int] = [random.choice(x) for x in state_actions]
        return self.next_state(states, actions_rand)

    @abstractmethod
    def get_state_actions(self, states: List[State]) -> List[List[Any]]:
        """ Get actions applicable to each states

        @param states: List of states
        @return: Applicable ations
        """
        pass

    @abstractmethod
    def sample_goal(self, states: List[State]) -> List[Goal]:
        pass

    @abstractmethod
    def is_solved(self, states: List[State], goals: List[Goal]) -> List[bool]:
        """ Returns whether or not state is solved

        @param states: List of states
        @param goals: List of goals
        @return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        pass

    @abstractmethod
    def get_start_states(self, num_states: int) -> List[State]:
        pass

    def get_start_goal_pairs(self, num_steps_l: List[int],
                             times: Optional[Times] = None) -> Tuple[List[State], List[Goal]]:
        # Initialize
        if times is None:
            times = Times()

        # Start states
        start_time = time.time()
        states_start: List[State] = self.get_start_states(len(num_steps_l))
        times.record_time("get_start_states", time.time() - start_time)

        # random walk
        start_time = time.time()
        states_goal: List[State] = self._random_walk(states_start, num_steps_l)
        times.record_time("random_walk", time.time() - start_time)

        # state to goal
        start_time = time.time()
        goals: List[Goal] = self.sample_goal(states_goal)
        times.record_time("sample_goal", time.time() - start_time)

        return states_start, goals

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[List[float]]]:
        """ Generate all children for the state
        @param states: List of states
        @return: Children of each state, Transition costs for each state
        """
        # TODO further validate
        # initialize
        states_exp_l: List[List[State]] = [[] for _ in range(len(states))]
        tcs_l: List[List[float]] = [[] for _ in range(len(states))]
        state_actions: List[List[Any]] = self.get_state_actions(states)

        num_actions_tot: np.array = np.array([len(x) for x in state_actions])
        num_actions_taken: np.array = np.zeros(len(states))
        actions_lt = num_actions_taken < num_actions_tot

        # for each move, get next states, transition costs, and if solved
        while np.any(actions_lt):
            idxs: np.ndarray = np.where(actions_lt)[0]
            states_idxs: List[State] = [states[idx] for idx in idxs]
            actions_idxs: List[Any] = [state_actions[idx].pop(0) for idx in idxs]

            # next state
            states_next, tcs_move = self.next_state(states_idxs, actions_idxs)

            # transition cost
            for exp_idx, idx in enumerate(idxs):
                states_exp_l[idx].append(states_next[exp_idx])
                tcs_l[idx].append(tcs_move[exp_idx])

            num_actions_taken[idxs] = num_actions_taken[idxs] + 1
            actions_lt[idxs] = num_actions_taken[idxs] < num_actions_tot[idxs]

        return states_exp_l, tcs_l

    @abstractmethod
    def states_to_nnet_input(self, states: List[State]) -> List[np.ndarray]:
        """ State to numpy arrays to be fed to the neural network

        @param states: List of states
        @return: List of numpy arrays. Each index along the first dimension of each array corresponds to the
        index of a state.
        """
        pass

    @abstractmethod
    def goals_to_nnet_input(self, goals: List[Goal]) -> List[np.ndarray]:
        """ Goals to numpy arrays to be fed to the neural network

        @param goals: List of goals
        @return: List of numpy arrays. Each index along the first dimension of each array corresponds to the
        index of a models.
        """
        pass

    @abstractmethod
    def get_v_nnet(self) -> HeurFnNNet:
        """ Get the neural network model for value iteration and A* search

        @return: neural network model
        """
        pass

    @abstractmethod
    def get_q_nnet(self) -> HeurFnNNet:
        """ Get the neural network model for Q-learning and Q* search

        @return: neural network model
        """
        pass

    @abstractmethod
    def get_pddl_domain(self) -> List[str]:
        """ Implement if using PDDL solvers, like fast-downward

        :return:
        """
        pass

    @abstractmethod
    def state_goal_to_pddl_inst(self, state: State, goal: Goal) -> List[str]:
        """ Implement if using PDDL solvers, like fast-downward

        :return:
        """
        pass

    @abstractmethod
    def pddl_action_to_action(self, pddl_action: str) -> int:
        """ Implement if using PDDL solvers, like fast-downward

        :return:
        """
        pass

    @abstractmethod
    def visualize(self, states: Union[List[State], List[Goal]]) -> np.ndarray:
        """ Implement if visualizing states

        :return:
        """
        pass

    def _random_walk(self, states: List[State], num_steps_l: List[int]) -> List[State]:
        states_walk: List[State] = [state for state in states]

        num_steps: np.array = np.array(num_steps_l)
        num_moves_curr: np.array = np.zeros(len(states))
        moves_lt = num_moves_curr < num_steps
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            states_moved, _ = self.next_state_rand(states_to_move)

            for move_idx, idx in enumerate(idxs):
                states_walk[idx] = states_moved[move_idx]

            num_moves_curr[idxs] = num_moves_curr[idxs] + 1

            moves_lt[idxs] = num_moves_curr[idxs] < num_steps[idxs]

        return states_walk


def get_preferred_model(states_start: List[State], goals_l: List[List[Model]],
                        heuristic_fn) -> Tuple[List[Model], np.array]:
    goals_pref: List[Model] = []

    set_lens: List[int] = [len(x) for x in goals_l]
    states_start_rep: List[List[State]] = [[] for _ in range(len(states_start))]
    goals_rep: List[List[Model]] = [[] for _ in range(len(states_start))]
    for set_len in range(1, max(set_lens) + 1):
        len_idxs = [i for i in range(len(goals_l)) if len(goals_l[i]) >= set_len]

        for len_idx in len_idxs:
            states_start_rep[len_idx].append(states_start[len_idx])
            goals_rep[len_idx].append(goals_l[len_idx][set_len - 1])

    states_start_rep_flat, split_idxs = misc_utils.flatten(states_start_rep)
    goals_rep_flat, _ = misc_utils.flatten(goals_rep)

    ctgs_rep_flat = heuristic_fn(states_start_rep_flat, goals_rep_flat).min(axis=1)
    ctgs_rep: List[List[float]] = misc_utils.unflatten(list(ctgs_rep_flat), split_idxs)
    ctgs_pref_l: List[float] = []
    for idx, ctg_rep in enumerate(ctgs_rep):
        min_idx: int = int(np.argmin(ctg_rep))
        ctgs_pref_l.append(ctg_rep[min_idx])
        goals_pref.append(goals_rep[idx][min_idx])

    return goals_pref, np.array(ctgs_pref_l)


class EnvGrndAtoms(Environment):
    def __init__(self, env_name: str):
        super().__init__(env_name)
        self.env_name: Optional[str] = env_name

    @abstractmethod
    def state_to_model(self, states: List[State]) -> List[Model]:
        pass

    @abstractmethod
    def model_to_state(self, models: List[Model]) -> List[State]:
        """ Assumes model is a fully specified state

        :param models:
        :return:
        """
        pass

    @abstractmethod
    def goal_to_model(self, goals: List[Goal]) -> List[Model]:
        pass

    @abstractmethod
    def model_to_goal(self, models: List[Model]) -> List[Goal]:
        pass

    @abstractmethod
    def start_state_fixed(self, states: List[State]) -> List[Model]:
        """ Given the start state, what must also be true for the goal state (i.e. immovable walls)

        :param states:
        :return:
        """
        pass

    def is_solved(self, states: List[State], goals: List[Goal]) -> List[bool]:
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

    def sample_goal(self, states: List[State]) -> List[Goal]:
        models_g: List[Model] = []

        models_s: List[Model] = self.state_to_model(states)
        keep_probs: np.array = np.random.rand(len(states))
        for model_s, keep_prob in zip(models_s, keep_probs):
            rand_subset: Set = misc_utils.random_subset(model_s, keep_prob)
            models_g.append(frozenset(rand_subset))

        return self.model_to_goal(models_g)

    @abstractmethod
    def get_ground_atoms(self) -> List[Atom]:
        """ Get all possible ground atoms that can be used to make a state

        :return:
        """
        pass

    @abstractmethod
    def on_model(self, m) -> Model:
        """ Process results from clingo

        :param m:
        :return:
        """
        pass

    @abstractmethod
    def get_bk(self) -> List[str]:
        """ get background, each element in list is a line

        :return:
        """
        pass
