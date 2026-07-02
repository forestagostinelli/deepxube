from typing import Tuple, Dict, Any

from deepxube.base.factory import Factory
from deepxube.base.domain import Domain
from deepxube.utils.command_line_utils import get_name_args

domain_factory: Factory[Domain] = Factory[Domain]("domain")


def get_domain_from_arg(domain: str) -> Tuple[Domain, str]:
    domain_name, domain_args = get_name_args(domain)
    domain_kwargs: Dict[str, Any] = domain_factory.get_kwargs(domain_name, domain_args)
    return domain_factory.build_class(domain_name, domain_kwargs), domain_name
