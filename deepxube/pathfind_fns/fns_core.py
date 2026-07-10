from dataclasses import dataclass
from deepxube.base.pathfind_fns import PFNsHeurV, PFNsHeurQ, PFNsPolicy, PFNsHeurVPolicy, PFNsHeurQPolicy
from deepxube.factories.pathfind_fns_factory import pathfind_fns_factory


@pathfind_fns_factory.register
@dataclass(frozen=True)
class FNsHeurVC(PFNsHeurV):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class FNsHeurQC(PFNsHeurQ):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsPolicyC(PFNsPolicy):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurVPolicyC(PFNsHeurVPolicy):
    pass


@pathfind_fns_factory.register
@dataclass(frozen=True)
class PFNsHeurQPolicyC(PFNsHeurQPolicy):
    pass
