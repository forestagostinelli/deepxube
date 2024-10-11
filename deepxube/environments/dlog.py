from typing import List, Tuple, Union, Set, Optional
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from deepxube.utils import misc_utils
import re
from random import randrange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel
from .environment_abstract import EnvGrndAtoms, State, Action, Goal, HeurFnNNet
from deepxube.logic.logic_objects import Atom, Model
from numpy.typing import NDArray

# setup our domain
# we choose the elliptic curve known as secp256k1
from ecpy.curves import Curve,Point
from ecpy.ecrand import rnd
curve: Curve = Curve.get_curve('secp256k1')

# Points on the curve, including the point at infinity, form a group.
# The order of the group can be accessed at curve.order.
# We will use additive notation for the group and capital letters for group elements.

# The generator point G represents our "solved" configuration.
# (G.x, G.y) = (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 , 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
G: Point = curve.generator


# Given a scalar k, a new point Q can be calculated as Q = k*G.
# However, we want to do it in reverse: given Q and G, we want to find k
# Therefore k is known as the discrete logarithm (dlog) of Q with respect to
# base point G.

# A State for us is simple. It is a point on the curve. As such it can be uniquely 
# represented by the bytes which represent the coordinates of the point on the curve.
# For our curve each point requires 512 bits = 64 bytes to represent the coordinates.
# We will use the conventional "uncompressed" representation of points which prepends
# the bytes with an additional byte of 0x04. So our state representation of a point
# is 65 bytes.
class DlogState(State):
    __slots__ = ['point', 'hash']

    def __init__(self, point: NDArray[np.uint8]):
        # uncompressed point on the curve (65 bytes)
        self.point: NDArray[np.uint8] = point
        self.hash: Optional[int] = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.point.tobytes())
        return self.hash
    
    def __eq__(self, other: object):
        if isinstance(other, DlogState):
            return np.array_equal(self.point, other.point)
        return NotImplemented

# A Goal for us is simply a target State     
class DlogGoal(Goal):
    def __init__(self, point: NDArray[np.uint8]):
        self.point: NDArray[np.uint8] = point

# There are two actions avilable for us:
#  0. addG -- given our current point P, obtain a new point P' by P' = P + G.
#  1. doubleP -- given our current point P, obtain a new point P' by P' = 2*P
# Reversing the addG action is easy enough: just subtract point G.
# Reversing the doubleP action requires multiplying our current point P by
#   the multiplicative inverse of 2 mod n where n is the curve order.
class DlogAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self):
        return self.action
    
    def __eq__(self, other: object):
        if isinstance(other, DlogAction):
            return self.action == other.action
        return NotImplemented

    
# verbatem from n_puzzle.py
class ProcessStates(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int):
        super().__init__()
        self.state_dim: int = state_dim
        self.one_hot_depth: int = one_hot_depth
        print("dlog one_hot_depth:",one_hot_depth)
        print("dlog state_dim: ", state_dim)

    def forward(self, states_nnet: Tensor):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        return x


# verbatim from n_puzzle.py
class FCResnet(nn.Module):
    def __init__(self, input_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int, out_dim: int,
                 batch_norm: bool, weight_norm: bool):
        super().__init__()
        self.first_fc = FullyConnectedModel(input_dim, [h1_dim, resnet_dim], [batch_norm] * 2, ["RELU"] * 2,
                                            weight_norms=[weight_norm] * 2)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, out_dim, batch_norm, weight_norm=weight_norm,
                                  layer_act="RELU")

    def forward(self, x: Tensor):
        x = self.first_fc(x)
        x = self.resnet(x)

        return x
    

# verbatim from n_puzzle.py
class NNet(HeurFnNNet):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_res_blocks: int,
                 out_dim: int, batch_norm: bool, weight_norm: bool, nnet_type: str):
        super().__init__(nnet_type)
        self.state_proc = ProcessStates(state_dim, one_hot_depth)

        input_dim: int = state_dim * one_hot_depth * 2
        self.heur = FCResnet(input_dim, h1_dim, resnet_dim, num_res_blocks, out_dim, batch_norm, weight_norm)

    def forward(self, states_goals_l: List[Tensor]):
        states_proc = self.state_proc(states_goals_l[0])
        goals_proc = self.state_proc(states_goals_l[1])

        x: Tensor = self.heur(torch.cat((states_proc, goals_proc), dim=1))

        return x
    

class Dlog(EnvGrndAtoms[DlogState, DlogAction, DlogGoal]):
    moves: List[str] = ['addG', 'doubleP']
    moves_rev: List[str] = ['subG', 'halveP']
    atomic_actions: List[str] = moves + moves_rev
    num_actions = len(atomic_actions)

    # not really sure what to put here for the state_dim which will be used by
    # the neural network. In the npuzzle and cube3 environments it seems related
    # to the size/difficulty of the puzzle/cube. It takes 65 bytes to represent
    # one of our states (which is an elliptic curve point in uncompressed form), so
    # we will start with that?
    state_dim: int = 65

    # a numpy representation of the byte 0x04 which is used when encoding/decoding points
    byte0x04_np: NDArray[np.uint8] = np.frombuffer(bytes.fromhex("04"), dtype=np.uint8)

    # sometimes we need to utilize the multiplicative inverse of two mod n
    # for secp256k1, the value is 57896044618658097711785492504343953926418782139537452191302581570759080747169
    inv2modn: int = pow(2,-1,curve.order)

    def __init__(self, env_name):
        super().__init__(env_name)

        # we are going to be representing things essentially as byte arrays
        self.dtype: type = np.uint8

        # Solved state
        self.goal_point: NDArray[np.uint8] = np.fromiter(curve.encode_point(G),dtype=np.uint8)

    def next_state(self, states: List[DlogState], actions: List[DlogAction]) -> Tuple[List[DlogState], List[float]]:
        # initialize
        states_np: NDArray[np.uint8] = np.stack([s.point for s in states], axis=0)
        
        states_next_np = np.zeros(states_np.shape, dtype=np.uint8)

        tcs_np: NDArray[np.float64] = np.zeros(len(states))
        for action in set(actions):
            action_idxs: NDArray[np.int_] = np.array([idx for idx in range(len(actions)) if actions[idx] == action])
            states_np_act = states_np[action_idxs]

            states_next_np_act = self._move_np(states_np_act, action.action)

            tcs_act: List[float] = [1.0 for _ in range(states_np_act.shape[0])]

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        states_next: List[DlogState] = [DlogState(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs
    
    def get_state_actions(self, states: List[DlogState]) -> List[List[DlogAction]]:
        return [[DlogAction(x) for x in range(self.num_actions)] for _ in range(len(states))]
    
    def is_solved(self, states: List[DlogState], goals: List[DlogGoal]) -> List[bool]:
        # note: this function might not actually need to be implemented? looks like there
        # is an implementation of it already in environment_abstract.py
        states_np = np.stack([x.point for x in states], axis=0)
        goals_np = np.stack([x.point for x in goals], axis=0)
        is_solved_np = np.all(states_np == goals_np, axis=1)
        return list(is_solved_np)
    
    def states_goals_to_nnet_input(self, states: List[DlogState], goals: List[DlogGoal]) -> List[NDArray[np.uint8]]:
        states_np: NDArray[np.uint8] = np.stack([state.point for state in states], axis=0).astype(np.uint8)
        goals_np: NDArray[np.uint8] = np.stack([goal.point for goal in goals], axis=0)
        return [states_np, goals_np]
    
    def state_to_model(self, states: List[DlogState]) -> List[Model]:
        # not sure what this should be for our environment
        raise NotImplementedError
    
    def model_to_state(self, states_m: List[Model]) -> List[DlogState]:
        # not sure what this should be for our environment
        raise NotImplementedError

    def goal_to_model(self, goals: List[DlogGoal]) -> List[Model]:
        # not sure what this should be for our environment
        raise NotImplementedError

    def model_to_goal(self, models: List[Model]) -> List[DlogGoal]:
        # not sure what this should be for our environment
        raise NotImplementedError
    
    def get_v_nnet(self) -> HeurFnNNet:
        # not sure what the appropriate state_dim is
        nnet = NNet(self.state_dim, 256, 5000, 1000, 4, 1, True, False, "V")
        return nnet

    def get_q_nnet(self) -> HeurFnNNet:
        # not sure what the appropriate state dim is
        # note: this line in n_puzzle.py has "V" for the network type but maybe it is an error? 
        #       It is changed to "Q" here to match what is in cube3.py 
        nnet = NNet(self.state_dim, 256, 5000, 1000, 4, self.num_actions, True, False, "Q")
        return nnet
    
    def get_start_states(self, num_states: int) -> List[DlogState]:
        """A method for generating start states. Should try to make this 
        generate states that are as diverse as possible so that the trained heuristic 
        function generalizes well."""
        assert (num_states > 0)
        states: List[DlogState] = []
        while len(states) < num_states:
            # choose a random scalar k less than the order of the curve
            k = rnd(curve.order)
            # construct our point P
            P = k*G
            P_bytes_np = np.fromiter(curve.encode_point(P), dtype=np.uint8)
            states.append(DlogState(P_bytes_np))

        return states
    
    def sample_goal(self, states_start: List[DlogState], states_goal: List[DlogState]) -> List[DlogGoal]:
        # FIXME: the below just returns the given states as goals
        goals: List[DlogGoal] = []
        for s in states_goal:
            goals.append(DlogGoal(s.point))
        return goals

        

    def start_state_fixed(self, states: List[DlogState]) -> List[Model]:
        return [frozenset() for _ in states]
    
    def get_pddl_domain(self) -> List[str]:
        raise NotImplementedError
    
    def state_goal_to_pddl_inst(self, state, goal) -> List[str]:
        raise NotImplementedError
    
    def pddl_action_to_action(self, pddl_action) -> int:
        raise NotImplementedError
    
    def visualize(self, states) -> NDArray[np.float64]:
        raise NotImplementedError
    
    def get_bk(self) -> List[str]:
        raise NotImplementedError
    
    def get_ground_atoms(self) -> List[Atom]:
        raise NotImplementedError
    
    def on_model(self, m) -> Model:
        raise NotImplementedError


    ## Domain-specific helper functions below
    #########
    
    def _addG_np(self, point_np: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # we assume point_np is in uncompressed form
        P: Point = curve.decode_point(point_np.tobytes())
        return np.fromiter(curve.encode_point(P + G),dtype=np.uint8)
    
    def _doubleP_np(self, point_np: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # we assume point_np is in uncompressed form
        P: Point = curve.decode_point(point_np.tobytes())
        return np.fromiter(curve.encode_point(2*P),dtype=np.uint8)
    
    def _subG_np(self, point_np: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # we assume point_np is in uncompressed form
        P: Point = curve.decode_point(point_np.tobytes())
        return np.fromiter(curve.encode_point(P - G),dtype=np.uint8)

    def _halveP_np(self, point_np: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # we assume point_np is in uncompressed form
        P: Point = curve.decode_point(point_np.tobytes())
        # multiplying P by the multiplicative inverse of 2 mod n
        return np.fromiter(curve.encode_point(self.inv2modn * P),dtype=np.uint8)

    def _move_np(self, point_np: NDArray[np.uint8], action: int) -> NDArray[np.uint8]:
        match (self.atomic_actions[action]):
            case "addG":
                return self._addG_np(point_np)
            case "subG":
                return self._subG_np(point_np)
            case "doubleP":
                return self._doubleP_np(point_np)
            case "halveP":
                return self._halveP_np(point_np)
            case _:
                return NotImplemented