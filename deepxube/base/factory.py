from typing import Dict, Any, Generic, TypeVar, Type, Callable, Optional, List
from abc import ABC, abstractmethod
import logging


class Parser(ABC):
    @abstractmethod
    def parse(self, args_str: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def help(self) -> str:
        pass


T = TypeVar("T")


class Factory(Generic[T]):
    def __init__(self, class_type_str: str):
        self._class_registry: Dict[str, Type[T]] = dict()
        self._parser_registry: Dict[str, Type[Parser]] = dict()
        self._class_type_str: str = class_type_str

    def register_class(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._class_registry.keys():
                raise ValueError(f"{self._class_type_str.capitalize()} {name!r} already registered")
            self._class_registry[name] = cls
            return cls
        return deco

    def register_parser(self, name: str) -> Callable[[Type[Parser]], Type[Parser]]:
        def deco(cls: Type[Parser]) -> Type[Parser]:
            if name in self._parser_registry.keys():
                raise ValueError(f"{self._class_type_str.capitalize()} parser {name!r} already registered")
            self._parser_registry[name] = cls
            return cls
        return deco

    def get_parser(self, name: str) -> Optional[Parser]:
        cls_parser: Optional[Type[Parser]] = self._parser_registry.get(name)
        if cls_parser is not None:
            return cls_parser()
        else:
            return None

    def get_kwargs(self, name: str, args_str: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict()
        parser: Optional[Parser] = self.get_parser(name)
        if (parser is not None) and (args_str is not None):
            try:
                kwargs = parser.parse(args_str)
            except Exception as e:
                logging.exception(f"Error occurred: {e}")
                raise ValueError(f"Error parsing {args_str} for {self._class_type_str} {name!r}.\nParser help:\n{parser.help()}")
        else:
            assert args_str is None, f"No parser for {self._class_type_str} {name}, however, args given are {args_str}"
        return kwargs

    def get_type(self, name: str) -> Type[T]:
        try:
            return self._class_registry[name]
        except KeyError:
            raise ValueError(f"Unknown {self._class_type_str} {name!r}. Available: {sorted(self._class_registry)}")

    def build_class(self, name: str, kwargs: Dict[str, Any]) -> T:
        cls: Type[T] = self.get_type(name)
        return cls(**kwargs)

    def get_all_class_names(self) -> List[str]:
        return list(self._class_registry.keys())
