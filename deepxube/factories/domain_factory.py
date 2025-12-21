from typing import Dict, Type, List, Optional, Any, Callable
from deepxube.base.domain import Domain, DomainParser
import logging


_domain_registry: Dict[str, Type[Domain]] = {}

_domain_parser_registry: Dict[str, Type[DomainParser]] = {}


def register_domain(domain_name: str) -> Callable[[Type[Domain]], Type[Domain]]:
    def deco(cls: Type[Domain]) -> Type[Domain]:
        if domain_name in _domain_registry.keys():
            raise ValueError(f"Domain {domain_name!r} already registered")
        _domain_registry[domain_name] = cls
        return cls
    return deco


def register_domain_parser(domain_name: str) -> Callable[[Type[DomainParser]], Type[DomainParser]]:
    def deco(cls: Type[DomainParser]) -> Type[DomainParser]:
        if domain_name in _domain_parser_registry.keys():
            raise ValueError(f"Domain parser {domain_name!r} already registered")
        _domain_parser_registry[domain_name] = cls
        return cls
    return deco


def get_domain_parser(domain_name: str) -> Optional[DomainParser]:
    if domain_name in _domain_parser_registry.keys():
        cls_parser: Type[DomainParser] = _domain_parser_registry[domain_name]
        parser: DomainParser = cls_parser()
        return parser
    else:
        return None


def get_domain_kwargs(domain_name: str, args_str: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict()
    parser: Optional[DomainParser] = get_domain_parser(domain_name)
    if (parser is not None) and (args_str is not None):
        try:
            kwargs = parser.parse(args_str)
        except Exception as e:
            logging.exception(f"Error occurred: {e}")
            raise ValueError(f"Error parsing {args_str} for domain {domain_name!r}. Help:\n{parser.help()}")
    else:
        assert args_str is None, (f"No parser for domain {domain_name}, however, args given are "
                                  f"{args_str}")
    return kwargs


def build_domain(domain_name: str, kwargs: Dict[str, Any]) -> Domain:
    try:
        cls: Type[Domain] = _domain_registry[domain_name]
    except KeyError:
        raise ValueError(
            f"Unknown domain {domain_name!r}. Available: {sorted(_domain_registry)}"
        )

    return cls(**kwargs)


def get_all_domain_names() -> List[str]:
    return list(_domain_registry.keys())


def get_domain_type(domain_name: str) -> type[Domain]:
    return _domain_registry[domain_name]
