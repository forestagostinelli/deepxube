from typing import Type, List

from deepxube.base.domain import Domain
from deepxube.base.pathfinding import PathFind
from deepxube.base.updater import Update, UpdateHER, UpdateHeur, UpdatePolicy, UpArgs
from deepxube.base.factory import Factory
from deepxube.utils.command_line_utils import get_pathfind_name_kwargs
from deepxube.factories.pathfinding_factory import pathfinding_factory


updater_factory: Factory[Update] = Factory[Update]("Update")


def get_updater(domain: Domain, pathfind_arg: str, up_args: UpArgs, her: bool, func_update: str) -> Update:
    up_cls_names: List[str] = updater_factory.get_all_class_names()
    pathfind_name: str = get_pathfind_name_kwargs(pathfind_arg)[0]
    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)

    up_cls_names = [up_cls_name for up_cls_name in up_cls_names if isinstance(domain, updater_factory.get_type(up_cls_name).domain_type())]
    up_cls_names = [up_cls_name for up_cls_name in up_cls_names if issubclass(pathfind_t, updater_factory.get_type(up_cls_name).pathfind_type())]
    up_cls_names = [up_cls_name for up_cls_name in up_cls_names if issubclass(updater_factory.get_type(up_cls_name), UpdateHER) == her]
    up_cls_names = [up_cls_name for up_cls_name in up_cls_names if updater_factory.get_type(up_cls_name).functions_type() is pathfind_t.functions_type()]

    if func_update.upper() == "HEUR":
        up_cls_names = [up_cls_name for up_cls_name in up_cls_names if issubclass(updater_factory.get_type(up_cls_name), UpdateHeur)]
    elif func_update.upper() == "POLICY":
        up_cls_names = [up_cls_name for up_cls_name in up_cls_names if issubclass(updater_factory.get_type(up_cls_name), UpdatePolicy)]
    else:
        raise ValueError(f"Unknown func to update {func_update}")

    if len(up_cls_names) == 0:
        raise ValueError(f"No updaters for Domain: {domain}, PathFind: {pathfind_t}, HER: {her}, Update func: {func_update}")
    if len(up_cls_names) > 1:
        raise ValueError(f"More than one updater option: {up_cls_names} for Domain: {domain}, PathFind: {pathfind_t}, HER: {her}, Update func: {func_update}")

    up_cls: Type[Update] = updater_factory.get_type(up_cls_names[0])
    return up_cls(domain, pathfind_arg, up_args)


"""
def get_updater(domain: Domain, heur_nnet_par: HeurNNetPar, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs,
                up_heur_args: UpHeurArgs, her: bool) -> UpdateHeur:
    # TODO how to handle backup and ub_parent_path for HER?

    pathfind_t: Type[PathFind] = pathfinding_factory.get_type(pathfind_name)
    if issubclass(pathfind_t, PathFindHasHeur):
        assert isinstance(domain, ActsEnum), (f"No updaters avaialable when doing reinforcement learning updates for domains that are not a suclass of "
                                              f"{ActsEnum}")
        updater_rl: UpdateHeurRL
        if isinstance(heur_nnet_par, HeurNNetParV):
            if not her:
                updater_rl = UpdateHeurVRLKeepGoal(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
            else:
                assert isinstance(domain, GoalSampleableFromState)
                updater_rl = UpdateHeurVRLHER(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
        elif isinstance(heur_nnet_par, HeurNNetParQ):
            if not her:
                updater_rl = UpdateHeurQRLKeepGoal(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
            else:
                assert isinstance(domain, GoalSampleableFromState)
                updater_rl = UpdateHeurQRLHER(domain, pathfind_name, pathfind_kwargs, up_args, up_heur_args)
        else:
            raise ValueError(f"No update implementation for {heur_nnet_par}")
        return updater_rl
    elif issubclass(pathfind_t, PathFindSup):
        assert her is False, "No hindsight experience replay (HER) for supervised learning"
        if isinstance(heur_nnet_par, HeurNNetParV):
            return UpdateHeurVSup(domain, pathfind_name, pathfind_kwargs, up_args)
        elif isinstance(heur_nnet_par, HeurNNetParQ):
            return UpdateHeurQSup(domain, pathfind_name, pathfind_kwargs, up_args)
        else:
            raise ValueError(f"No update implementation for {heur_nnet_par}")
    else:
        raise ValueError(f"Unknown update method for {pathfind_t}")
"""
