from typing import Any, Dict, Tuple, Type, List, Optional

import torch
from torch import nn

from deepxube.utils.command_line_utils import get_name_args
from deepxube.pytorch.nnet_utils import NNetCallable, load_nnet
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.nnet_factory import deepxube_nnet_factory
from deepxube.base.domain import Domain
from deepxube.base.nnet_input import NNetInput
from deepxube.base.nnet import DeepXubeNNet
from deepxube.base.pathfind_fns import PFNs, DeepXubeNNetPar
from deepxube.base.factory import FactoryAutoBuild, Factory

deepxube_nnet_par_factory: Factory[DeepXubeNNetPar] = Factory[DeepXubeNNetPar]("DeepXubeNNetPar")

pathfind_fns_factory: FactoryAutoBuild[PFNs] = FactoryAutoBuild[PFNs]("PathFindFNs")


def get_dx_nnet_par(domain: Domain, domain_name: str, nnet_par_name_args: str, nnet_name_args: str) -> Tuple[DeepXubeNNetPar, str]:
    nnet_par_name, nnet_par_args = get_name_args(nnet_par_name_args)
    nnet_name, nnet_args = get_name_args(nnet_name_args)

    nnet_par_t: Type[DeepXubeNNetPar] = deepxube_nnet_par_factory.get_type(nnet_par_name)
    nnet_t: Type[DeepXubeNNet] = deepxube_nnet_factory.get_type(nnet_name)

    # find nnet input and create class
    incompat_reasons: List[str] = []
    nnet_input_domain_names: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
    for nnet_input_domain_name in nnet_input_domain_names:
        nnet_input_domain_t: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_name)
        incompat_reason: Optional[str] = nnet_par_t.get_incompat_reason(domain, nnet_input_domain_t, nnet_t)
        if incompat_reason is not None:
            incompat_reasons.append(incompat_reason + f" (NNetInput name: {nnet_input_domain_name})")
        else:
            nnet_par_kwargs: Dict[str, Any] = deepxube_nnet_par_factory.get_kwargs(nnet_par_name, nnet_par_args)
            nnet_par_kwargs["domain"] = domain
            nnet_par_kwargs["nnet_input_name"] = nnet_input_domain_name
            nnet_par_kwargs["nnet_name"] = nnet_name
            nnet_par_kwargs["nnet_kwargs"] = deepxube_nnet_factory.get_kwargs(nnet_name, nnet_args)

            return deepxube_nnet_par_factory.build_class(nnet_par_name, nnet_par_kwargs), nnet_par_name

    incompat_reasons_str: str = '\n'.join(incompat_reasons)
    raise ValueError(f"Cannot build fn for domain: {domain_name}, nnet_args: {nnet_name_args}, nnet_par_args: {nnet_par_name_args}."
                     f"\nIncompatibility reasons:\n{incompat_reasons_str}")


def get_fn_dicts(domain: Domain, domain_name: str, fn_name_args_l: List[str], device: torch.device, nnet_files: Optional[List[Optional[str]]] = None,
                 nnet_batch_size: Optional[int] = None) -> Tuple[Dict[str, NNetCallable], Dict[str, DeepXubeNNetPar]]:
    nnet_fn_dict: Dict[str, NNetCallable] = dict()
    nnet_par_dict: Dict[str, DeepXubeNNetPar] = dict()
    if nnet_files is not None:
        assert len(nnet_files) == len(fn_name_args_l)

    for fn_idx, fn_arg in enumerate(fn_name_args_l):
        # get nnet par
        fn_arg_split: List[str] = fn_arg.split(",")
        assert len(fn_arg_split) == 2, f"{fn_arg} split len != 2 when splitting on comma"
        nnet_par_name_args, nnet_name_args = fn_arg_split[0], fn_arg_split[1]
        nnet_par, nnet_par_name = get_dx_nnet_par(domain, domain_name, nnet_par_name_args, nnet_name_args)

        # get nnet
        nnet: nn.Module = nnet_par.get_nnet()
        if nnet_files is not None:
            nnet_file: Optional[str] = nnet_files[fn_idx]
            if nnet_file is None:
                nnet_par.set_use_default_fn(True)
            else:
                nnet_par.set_nnet_file(nnet_file)

                nnet: nn.Module = load_nnet(nnet_file, nnet)
                nnet.eval()
                nnet.to(device)
                nnet = nn.DataParallel(nnet)

        field_name: str = nnet_par.get_field_name()
        nnet_fn_dict[field_name] = nnet_par.get_nnet_fn(nnet, nnet_batch_size, device, None)
        nnet_par_dict[field_name] = nnet_par

    return nnet_fn_dict, nnet_par_dict
