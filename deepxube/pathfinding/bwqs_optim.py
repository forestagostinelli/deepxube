from typing import List, Optional, Type, Any, Dict

from deepxube.base.domain import ActsOptim, State, Goal, Action
from deepxube.factories.pathfinding_factory import pathfinding_factory
from deepxube.pathfinding.bwqs import BWQSActsAny
from deepxube.base.factory import Parser


class BWQSActsOptim(BWQSActsAny[ActsOptim]):
    """Batch Weighted Q* Search with domain-driven action proposals.

    The domain provides candidate actions via ``ActsOptim.get_state_actions_opt`` which
    can leverage the current heuristic ``h(s, g, a)`` to search effectively in
    continuous or very large action spaces.
    """

    def __init__(self, domain: ActsOptim, batch_size: int = 1, weight: float = 1.0, eps: float = 0.0,
                 num_actions: Optional[int] = None):
        super().__init__(domain, batch_size=batch_size, weight=weight, eps=eps)
        self.num_actions: Optional[int] = num_actions

    @staticmethod
    def domain_type() -> Type[ActsOptim]:
        return ActsOptim

    def _get_actions(self, states: List[State], goals: List[Goal]) -> List[List[Action]]:
        assert self.heur_fn is not None, "Heuristic function must be set before calling _get_actions"
        return self.domain.get_state_actions_opt(states, goals, self.heur_fn, num_actions=self.num_actions)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(batch_size={self.batch_size_default}, weight={self.weight_default}, "
                f"eps={self.eps_default}, num_actions={self.num_actions})")


@pathfinding_factory.register_parser("bwqs_optim")
class BWQSActsOptimParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        """Parse strings of form 'batch_weight_eps_numacts'.

        numacts can be 'None' to pass through without limiting.
        """
        args = args_str.split("_")
        assert len(args) == 4, "Expected format 'batch_weight_eps_numacts'"
        batch_size = int(args[0])
        weight = float(args[1])
        eps = float(args[2])
        num_actions = None if args[3].lower() == "none" else int(args[3])
        return {"batch_size": batch_size, "weight": weight, "eps": eps, "num_actions": num_actions}

    def help(self) -> str:
        return "Batch size, weight, eps, num_actions (or None). Example: 'bwqs_optim.8_1.2_0.1_16'"


@pathfinding_factory.register_class("bwqs_optim")
class BWQSActsOptimRegistered(BWQSActsOptim):
    pass
