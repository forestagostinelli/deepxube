from typing import List, Tuple
from deepxube.base.domain import Domain, State, Action, Goal
from deepxube.base.pathfind_fns import HeurFn, HeurVFn, HeurQFn, PolicyFn
from deepxube.utils import misc_utils


def get_zero_heur(heur_type: str) -> HeurFn:
    heur_fn: HeurFn
    if heur_type.upper() == "V":
        class HeurZerosVFn(HeurVFn):
            def __call__(self, states_in: List[State], goals_in: List[Goal]) -> List[float]:
                return [0.0] * len(states_in)

        heur_fn = HeurZerosVFn()
    elif heur_type.upper() in {"QFIX", "QIN"}:
        class HeurZerosQFn(HeurQFn):
            def __call__(self, states_in: List[State], goals_in: List[Goal], actions_l_in: List[List[Action]]) -> List[List[float]]:
                heur_vals_l: List[List[float]] = []
                for actions_in in actions_l_in:
                    heur_vals_l.append([0.0] * len(actions_in))
                return heur_vals_l

        heur_fn = HeurZerosQFn()
    else:
        raise ValueError(f"Unknown heur type {heur_type}")

    return heur_fn


def policy_fn_rand(domain: Domain, states: List[State], num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
    if num_rand == 0:
        return [[] for _ in states], [[] for _ in states]

    states_rep: List[List[State]] = []
    for state in states:
        states_rep.append([state] * num_rand)

    states_rep_flat, split_idxs = misc_utils.flatten(states_rep)

    actions_samp_flat: List[Action] = domain.sample_state_action(states_rep_flat)
    actions_samp_l: List[List[Action]] = misc_utils.unflatten(actions_samp_flat, split_idxs)

    probs_l: List[List[float]] = []
    for actions_samp_i in actions_samp_l:
        probs_l.append([1.0 / len(actions_samp_i)] * len(actions_samp_i))

    return actions_samp_l, probs_l


def get_rand_policy(domain: Domain, policy_samp: int) -> PolicyFn:
    class PolicyFnRand(PolicyFn):
        def __call__(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
            return policy_fn_rand(domain, states, policy_samp)

    return PolicyFnRand()
