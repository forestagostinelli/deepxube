from deepxube.base.pathfind_fns import PFNs
from deepxube.base.factory import FactoryAutoBuild

pathfind_fns_factory: FactoryAutoBuild[PFNs] = FactoryAutoBuild[PFNs]("PathFindFNs")
